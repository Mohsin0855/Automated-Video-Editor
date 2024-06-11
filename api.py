from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.param_functions import File, Form
import os
from typing import List
from werkzeug.utils import secure_filename
import subprocess
from scipy.io import wavfile
import numpy as np
import math
from shutil import copyfile, rmtree
from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter
import cv2
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from whisper import load_model
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import torch
import numpy as np
import shutil
import librosa
import soundfile as sf
import openai

app = FastAPI()
os.environ['UPLOAD_FOLDER'] = 'uploaded_videos'
os.environ['MAX_CONTENT_LENGTH'] = '16777216'  # 16 * 1024 * 1024
openai.api_key = "" # Add you openai api here
os.makedirs(os.environ['UPLOAD_FOLDER'], exist_ok=True)

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()


def audio_trimmer_function(input_file, output_file, silent_threshold=0.03, sounded_speed=1, silent_speed=999999, frame_margin=2, sample_rate=44100, frame_rate=30, frame_quality=3):
    TEMP_FOLDER = "temp_audio_trimmer"

    def getMaxVolume(s):
        return float(np.max(s))

    def copyFrame(inputFrame, outputFrame):
        src = f"{TEMP_FOLDER}/frame{inputFrame + 1:06d}.jpg"
        dst = f"{TEMP_FOLDER}/newFrame{outputFrame + 1:06d}.jpg"
        if not os.path.isfile(src):
            return False
        copyfile(src, dst)
        if outputFrame % 20 == 19:
            print(str(outputFrame+1)+" time-altered frames saved.")
        return True

    def createPath(s):
        try:
            os.mkdir(s)
        except OSError:
            assert False, "Creation of the directory %s failed. (The TEMP folder may already exist. Delete or rename it, and try again.)"

    def deletePath(s):
        try:
            rmtree(s, ignore_errors=False)
        except OSError:
            print("Deletion of the directory %s failed" % s)

    frameRate = 30
    SAMPLE_RATE = 44100
    SILENT_THRESHOLD = silent_threshold
    FRAME_SPREADAGE = frame_margin
    NEW_SPEED = [silent_speed, sounded_speed]
    FRAME_QUALITY = 3
    TEMP_FOLDER = "TEMP"
    AUDIO_FADE_ENVELOPE_SIZE = 100

    createPath(TEMP_FOLDER)
    
    commands = [
        f"ffmpeg -i {input_file} {TEMP_FOLDER}/frame%06d.jpg -hide_banner",
        f"ffmpeg -i {input_file} -ab 160k -ac 2 -ar {SAMPLE_RATE} -vn {TEMP_FOLDER}/audio.wav"
    ]

    for command in commands:
        subprocess.call(command, shell=True)

    sampleRate, audioData = wavfile.read(f"{TEMP_FOLDER}/audio.wav")
    audioSampleCount = audioData.shape[0]
    maxAudioVolume = getMaxVolume(audioData)

    samplesPerFrame = sampleRate/frameRate
    audioFrameCount = int(math.ceil(audioSampleCount/samplesPerFrame))
    hasLoudAudio = np.zeros((audioFrameCount))

    for i in range(audioFrameCount):
        start = int(i*samplesPerFrame)
        end = min(int((i+1)*samplesPerFrame), audioSampleCount)
        audiochunks = audioData[start:end]
        maxchunksVolume = float(getMaxVolume(audiochunks))/maxAudioVolume
        if maxchunksVolume >= SILENT_THRESHOLD:
            hasLoudAudio[i] = 1

    chunks = [[0, 0, 0]]
    shouldIncludeFrame = np.zeros((audioFrameCount))
    for i in range(audioFrameCount):
        start = int(max(0, i-FRAME_SPREADAGE))
        end = int(min(audioFrameCount, i+1+FRAME_SPREADAGE))
        shouldIncludeFrame[i] = np.max(hasLoudAudio[start:end])
        if (i >= 1 and shouldIncludeFrame[i] != shouldIncludeFrame[i-1]):
            chunks.append([chunks[-1][1], i, shouldIncludeFrame[i-1]])

    chunks.append([chunks[-1][1], audioFrameCount, shouldIncludeFrame[i-1]])
    chunks = chunks[1:]

    outputAudioData = np.zeros((0, audioData.shape[1]))
    outputPointer = 0

    lastExistingFrame = None
    for chunk in chunks:
        audioChunk = audioData[int(chunk[0]*samplesPerFrame):int(chunk[1]*samplesPerFrame)]

        sFile = f"{TEMP_FOLDER}/tempStart.wav"
        eFile = f"{TEMP_FOLDER}/tempEnd.wav"
        wavfile.write(sFile, SAMPLE_RATE, audioChunk)
        with WavReader(sFile) as reader:
            with WavWriter(eFile, reader.channels, reader.samplerate) as writer:
                tsm = phasevocoder(reader.channels, speed=NEW_SPEED[int(chunk[2])])
                tsm.run(reader, writer)
        _, alteredAudioData = wavfile.read(eFile)
        leng = alteredAudioData.shape[0]
        endPointer = outputPointer+leng
        outputAudioData = np.concatenate((outputAudioData, alteredAudioData/maxAudioVolume))

        if leng < AUDIO_FADE_ENVELOPE_SIZE:
            outputAudioData[outputPointer:endPointer] = 0
        else:
            premask = np.arange(AUDIO_FADE_ENVELOPE_SIZE)/AUDIO_FADE_ENVELOPE_SIZE
            mask = np.repeat(premask[:, np.newaxis], 2, axis=1)
            outputAudioData[outputPointer:outputPointer+AUDIO_FADE_ENVELOPE_SIZE] *= mask
            outputAudioData[endPointer-AUDIO_FADE_ENVELOPE_SIZE:endPointer] *= 1-mask

        startOutputFrame = int(math.ceil(outputPointer/samplesPerFrame))
        endOutputFrame = int(math.ceil(endPointer/samplesPerFrame))
        for outputFrame in range(startOutputFrame, endOutputFrame):
            inputFrame = int(chunk[0]+NEW_SPEED[int(chunk[2])]*(outputFrame-startOutputFrame))
            didItWork = copyFrame(inputFrame, outputFrame)
            if didItWork:
                lastExistingFrame = inputFrame
            else:
                copyFrame(lastExistingFrame, outputFrame)

        outputPointer = endPointer

    wavfile.write(f"{TEMP_FOLDER}/audioNew.wav", SAMPLE_RATE, outputAudioData)

    command = f"ffmpeg -framerate {frameRate} -i {TEMP_FOLDER}/newFrame%06d.jpg -i {TEMP_FOLDER}/audioNew.wav -strict -2 {output_file}"
    subprocess.call(command, shell=True)

    deletePath(TEMP_FOLDER)

    return output_file

def caption_generator_function(video_path, output_video_path):
    model = load_model("base")
    audio_path = os.path.join(os.path.dirname(video_path), "audio.mp3")
    video_clip = VideoFileClip(video_path)
    audio = video_clip.audio
    audio.write_audiofile(audio_path)

    result = model.transcribe(audio_path)
    text_array = []
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    for segment in result["segments"]:
        start, end, text = segment["start"], segment["end"], segment["text"]
        total_frames = int((end - start) * fps)
        start_frame = int(start * fps)
        words = text.split()
        current_text = ""
        current_length = 0
        frame_texts = []

        for word in words:
            if current_length + len(word) * 12 < width:
                current_text += word + " "
                current_length += len(word) * 12
            else:
                frame_texts.append((current_text, start_frame, start_frame + total_frames))
                current_text = word + " "
                current_length = len(word) * 12
                start_frame += total_frames

        if current_text:
            frame_texts.append((current_text, start_frame, start_frame + total_frames))

        text_array.extend(frame_texts)

    output_folder = os.path.join(os.path.dirname(video_path), "frames")
    os.makedirs(output_folder, exist_ok=True)
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for text, start_frame, end_frame in text_array:
            if start_frame <= frame_number <= end_frame:
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                break

        cv2.imwrite(os.path.join(output_folder, f"{frame_number}.jpg"), frame)
        frame_number += 1

    cap.release()

    images = sorted(os.listdir(output_folder), key=lambda x: int(x.split(".")[0]))
    frames = [cv2.imread(os.path.join(output_folder, image)) for image in images]
    clip = ImageSequenceClip([os.path.join(output_folder, image) for image in images], fps=fps)
    audio_clip = AudioFileClip(audio_path)
    final_clip = clip.set_audio(audio_clip)
    final_clip.write_videofile(output_video_path)

    video_clip.close()
    audio_clip.close()
    final_clip.close()

    return output_video_path

def video_summarization_function(video_file_path, outputs_dir, segment_length=600):
    raw_audio_dir = os.path.join(outputs_dir, "raw_audio")
    chunks_dir = os.path.join(outputs_dir, "chunks")
    transcripts_file = os.path.join(outputs_dir, "transcripts.txt")
    summary_file = os.path.join(outputs_dir, "summary.txt")

    if os.path.exists(outputs_dir):
        shutil.rmtree(outputs_dir)
    os.makedirs(outputs_dir)
    os.makedirs(raw_audio_dir)
    os.makedirs(chunks_dir)

    def chunk_audio(video_file_path, segment_length, output_dir):
        with VideoFileClip(video_file_path) as video_clip:
            audio_path = os.path.join(raw_audio_dir, "audio.wav")
            video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')

        audio, sr = librosa.load(audio_path, sr=44100)
        duration = librosa.get_duration(y=audio, sr=sr)
        num_segments = int(duration / segment_length) + (duration % segment_length > 0)

        for i in range(num_segments):
            start = i * segment_length * sr
            end = (i + 1) * segment_length * sr
            segment = audio[start:end]
            sf.write(os.path.join(output_dir, f"segment_{i}.wav"), segment, sr)
        
        return [os.path.join(output_dir, f"segment_{i}.wav") for i in range(num_segments)]

    chunked_audio_files = chunk_audio(video_file_path, segment_length, chunks_dir)

    def transcribe_audio(audio_files):
        transcripts = []
        for audio_file in audio_files:
            with open(audio_file, "rb") as audio:
                response = openai.Audio.transcribe("whisper-1", audio)
                transcripts.append(response["text"])

        with open(transcripts_file, "w") as file:
            for transcript in transcripts:
                file.write(transcript + "\n")

        return transcripts

    transcriptions = transcribe_audio(chunked_audio_files)

    def summarize(transcriptions):
        system_prompt = """
        You are a helpful assistant that summarizes video content. Summarize the key points to one or two sentences that capture the essence of the video.
        """
        long_summary = "\n".join(transcriptions)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": long_summary},
            ],
        )
        summary = response["choices"][0]["message"]["content"]
        with open(summary_file, "w") as file:
            file.write(summary)

        return summary

    short_summary = summarize(transcriptions)

    return short_summary

@app.post('/process-video')
async def process_video(video: UploadFile = File(...), process_steps: List[str] = Form(...)):
    if video.filename == '':
        raise HTTPException(status_code=400, detail='No selected file')
    
    filename = secure_filename(video.filename)
    filepath = os.path.join(os.environ['UPLOAD_FOLDER'], filename)

    with open(filepath, "wb") as buffer:
        buffer.write(await video.read())

    output = filepath
    output_dir = r"uploaded_videos\output"

    if 'trim' in process_steps:
        output = audio_trimmer_function(filepath, output)
    if 'caption' in process_steps:
        output = caption_generator_function(filepath, output)
    if 'summarize' in process_steps:
        summary = video_summarization_function(filepath, output_dir)
        return {"message": "Video processed", "summary": summary, "output": output}

    return {"message": "Video processed without summarization", "output": output}

@app.post('/generate_video')
async def generate_video(request: Request):
    data = await request.json()
    prompt = data.get('prompt', '')
    if not prompt:
        raise HTTPException(status_code=400, detail='Prompt not provided')
    
    output_dir = r"generated_videos"    
    video_frames = pipe(prompt, num_inference_steps=25).frames
    video_frames_np = [np.array(frame) for frame in video_frames]
    video_frames_np = np.concatenate(video_frames_np, axis=0)
    video_path = export_to_video(video_frames_np, output_video_path=os.path.join(output_dir, "output.mp4"))
    
    return {'video_path': video_path}

@app.route('/')
async def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
