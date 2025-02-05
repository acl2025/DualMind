# # whisper_STT.py

global_whisper_model = None

def init_stt_worker():
    global global_whisper_model
    print("Loading Whisper model...")
    # 在子进程中导入，避免主进程初始化 CUDA
    import whisper
    global_whisper_model = whisper.load_model("large")

def stt_worker(audio_chunk, previous_text):
    global global_whisper_model
    # 在函数内部引用 whisper，确保在子进程中使用
    whisper_result = global_whisper_model.transcribe(
        audio_chunk,
        language='zh',
        task='transcribe',
        no_speech_threshold=0.05,
        logprob_threshold=-1.0,
        condition_on_previous_text=False,
        #initial_prompt=previous_text,
        # 添加以下配置来处理特殊token
        suppress_tokens=[-1],  # 抑制大多数特殊token
        # allowed_special={'<|pa|>'},  # 添加这行来允许这个特殊token
    )
    return whisper_result





# global_whisper_model = None

# def init_stt_worker():
#     global global_whisper_model
#     print("Loading Whisper model...")
#     # 在子进程中导入，避免主进程初始化 CUDA
#     import whisper
#     global_whisper_model = whisper.load_model("large")

# def stt_worker(audio_chunk):
#     global global_whisper_model
#     # 在函数内部引用 whisper，确保在子进程中使用
#     whisper_result = global_whisper_model.transcribe(
#         audio_chunk,
#         language='zh',
#         task='transcribe',
#         without_timestamps=True
#     )
#     return whisper_result