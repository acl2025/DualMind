#!/usr/bin/env python3
import sys
import numpy as np
import librosa
from functools import lru_cache
import time
import logging

import io
import soundfile as sf
import math

logger = logging.getLogger(__name__)

@lru_cache(10**6)
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s]

# Whisper backend

class ASRBase:

    sep = " "   # join transcribe words with this character (" " for whisper_timestamped,
                # "" for faster-whisper because it emits the spaces when needed)

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.logfile = logfile

        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan

        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize, cache_dir):
        raise NotImplemented("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplemented("must be implemented in the child class")

    def use_vad(self):
        raise NotImplemented("must be implemented in the child class")


class WhisperTimestampedASR(ASRBase):
    """Uses whisper_timestamped library as the backend."""

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        import whisper
        import whisper_timestamped
        from whisper_timestamped import transcribe_timestamped
        self.transcribe_timestamped = transcribe_timestamped
        if model_dir is not None:
            logger.debug("ignoring model_dir, not implemented")
        return whisper.load_model(modelsize, download_root=cache_dir)

    def transcribe(self, audio, init_prompt=""):
        result = self.transcribe_timestamped(self.model,
                audio, language=self.original_language,
                initial_prompt=init_prompt, verbose=None,
                condition_on_previous_text=True, **self.transcribe_kargs)
        return result

    def ts_words(self,r):
        # return: transcribe result object to [(beg,end,"word1"), ...]
        o = []
        for s in r["segments"]:
            for w in s["words"]:
                t = (w["start"],w["end"],w["text"])
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s["end"] for s in res["segments"]]

    def use_vad(self):
        self.transcribe_kargs["vad"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class FasterWhisperASR(ASRBase):
    """Uses faster-whisper library as the backend."""

    sep = ""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel
        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")

        model = WhisperModel(model_size_or_path, device="cuda", compute_type="float16", download_root=cache_dir)
        return model

    def transcribe(self, audio, init_prompt=""):
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs
        )
        return list(segments)

    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class MLXWhisper(ASRBase):
    """
    Uses MLX Whisper library as the backend, optimized for Apple Silicon.
    """

    sep = " "

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from mlx_whisper.transcribe import ModelHolder, transcribe
        import mlx.core as mx
        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize parameter is not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = self.translate_model_name(modelsize)
            logger.debug(f"Loading whisper model {modelsize}. You use mlx whisper, so {model_size_or_path} will be used.")
        self.model_size_or_path = model_size_or_path

        dtype = mx.float16
        ModelHolder.get_model(model_size_or_path, dtype)
        return transcribe

    def translate_model_name(self, model_name):
        model_mapping = {
            "tiny.en": "mlx-community/whisper-tiny.en-mlx",
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base.en": "mlx-community/whisper-base.en-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small.en": "mlx-community/whisper-small.en-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium.en": "mlx-community/whisper-medium.en-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large-v1": "mlx-community/whisper-large-v1-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
            "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
            "large": "mlx-community/whisper-large-mlx"
        }
        mlx_model_path = model_mapping.get(model_name)
        if mlx_model_path:
            return mlx_model_path
        else:
            raise ValueError(f"Model name '{model_name}' is not recognized or not supported.")

    def transcribe(self, audio, init_prompt=""):
        segments = self.model(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            word_timestamps=True,
            condition_on_previous_text=True,
            path_or_hf_repo=self.model_size_or_path,
            **self.transcribe_kargs
        )
        return segments.get("segments", [])

    def ts_words(self, segments):
        return [
            (word["start"], word["end"], word["word"])
            for segment in segments
            for word in segment.get("words", [])
            if segment.get("no_speech_prob", 0) <= 0.9
        ]

    def segments_end_ts(self, res):
        return [s['end'] for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class OpenaiApiASR(ASRBase):
    """Uses OpenAI's Whisper API for audio transcription."""

    def __init__(self, lan=None, temperature=0, logfile=sys.stderr):
        self.logfile = logfile
        self.modelname = "whisper-1"
        self.original_language = None if lan == "auto" else lan
        self.response_format = "verbose_json"
        self.temperature = temperature

        self.load_model()
        self.use_vad_opt = False
        self.task = "transcribe"

    def load_model(self, *args, **kwargs):
        from openai import OpenAI
        self.client = OpenAI()
        self.transcribed_seconds = 0

    def ts_words(self, segments):
        no_speech_segments = []
        if self.use_vad_opt:
            for segment in segments.segments:
                if segment["no_speech_prob"] > 0.8:
                    no_speech_segments.append((segment.get("start"), segment.get("end")))

        o = []
        for word in segments.words:
            start = word.start
            end = word.end
            if any(s[0] <= start <= s[1] for s in no_speech_segments):
                continue
            o.append((start, end, word.word))
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res.words]

    def transcribe(self, audio_data, prompt=None, *args, **kwargs):
        buffer = io.BytesIO()
        buffer.name = "temp.wav"
        sf.write(buffer, audio_data, samplerate=16000, format='WAV', subtype='PCM_16')
        buffer.seek(0)

        self.transcribed_seconds += math.ceil(len(audio_data)/16000)

        params = {
            "model": self.modelname,
            "file": buffer,
            "response_format": self.response_format,
            "temperature": self.temperature,
            "timestamp_granularities": ["word", "segment"]
        }
        if self.task != "translate" and self.original_language:
            params["language"] = self.original_language
        if prompt:
            params["prompt"] = prompt

        if self.task == "translate":
            proc = self.client.audio.translations
        else:
            proc = self.client.audio.transcriptions

        transcript = proc.create(**params)
        logger.debug(f"OpenAI API processed accumulated {self.transcribed_seconds} seconds")

        return transcript

    def use_vad(self):
        self.use_vad_opt = True

    def set_translate_task(self):
        self.task = "translate"


class HypothesisBuffer:
    def __init__(self, logfile=sys.stderr):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_commited_time = 0
        self.last_commited_word = None

        self.logfile = logfile

    def insert(self, new, offset):
        new = [(a+offset,b+offset,t) for a,b,t in new]
        self.new = [(a,b,t) for a,b,t in new if a > self.last_commited_time-0.1]

        if len(self.new) >= 1:
            a,b,t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1,min(min(cn,nn),5)+1):
                        c = " ".join([self.commited_in_buffer[-j][2] for j in range(1,i+1)][::-1])
                        tail = " ".join(self.new[j-1][2] for j in range(1,i+1))
                        if c == tail:
                            words = []
                            for j in range(i):
                                words.append(repr(self.new.pop(0)))
                            words_msg = " ".join(words)
                            logger.debug(f"removing last {i} words: {words_msg}")
                            break

    def flush(self):
        commit = []
        while self.new:
            na, nb, nt = self.new[0]
            if len(self.buffer) == 0:
                break
            if nt == self.buffer[0][2]:
                commit.append((na,nb,nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer


class OnlineASRProcessor:
    SAMPLING_RATE = 16000

    def __init__(self, asr, tokenizer=None, buffer_trimming=("segment", 15), logfile=sys.stderr):
        """
        asr: WhisperASR object
        tokenizer: 如果 buffer_trimming='sentence' 时需要用来分句的 tokenizer
        buffer_trimming: ('segment', 15) 或 ('sentence', 15)
        """
        self.asr = asr
        self.tokenizer = tokenizer
        self.logfile = logfile

        self.init()

        self.buffer_trimming_way, self.buffer_trimming_sec = buffer_trimming

    def init(self, offset=None):
        self.audio_buffer = np.array([],dtype=np.float32)
        self.transcript_buffer = HypothesisBuffer(logfile=self.logfile)
        self.buffer_time_offset = 0
        if offset is not None:
            self.buffer_time_offset = offset
        self.transcript_buffer.last_commited_time = self.buffer_time_offset
        self.commited = []

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        k = max(0,len(self.commited)-1)
        while k > 0 and self.commited[k-1][1] > self.buffer_time_offset:
            k -= 1

        p = self.commited[:k]
        p = [t for _,_,t in p]
        prompt = []
        l = 0
        while p and l < 200:
            x = p.pop(-1)
            l += len(x)+1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(t for _,_,t in non_prompt)

    def process_iter(self):
        prompt, non_prompt = self.prompt()
        logger.debug(f"PROMPT: {prompt}")
        logger.debug(f"CONTEXT: {non_prompt}")
        logger.debug(f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}")
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)

        tsw = self.asr.ts_words(res)
        
        #print(tsw[0][2])
        
        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)
        completed = self.to_flush(o)
        logger.debug(f">>>>COMPLETE NOW: {completed}")
        the_rest = self.to_flush(self.transcript_buffer.complete())
        logger.debug(f"INCOMPLETE: {the_rest}")

        if o and self.buffer_trimming_way == "sentence":
            if len(self.audio_buffer)/self.SAMPLING_RATE > self.buffer_trimming_sec:
                self.chunk_completed_sentence()

        if self.buffer_trimming_way == "segment":
            s = self.buffer_trimming_sec
        else:
            s = 30
        if len(self.audio_buffer)/self.SAMPLING_RATE > s:
            self.chunk_completed_segment(res)

        logger.debug(f"len of buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}")
        return self.to_flush(o)

    def chunk_completed_sentence(self):
        if self.commited == []: return
        logger.debug(self.commited)
        sents = self.words_to_sentences(self.commited)
        for s in sents:
            logger.debug(f"\t\tSENT: {s}")
        if len(sents) < 2:
            return
        while len(sents) > 2:
            sents.pop(0)
        chunk_at = sents[-2][1]

        logger.debug(f"--- sentence chunked at {chunk_at:2.2f}")
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        if self.commited == []: return
        ends = self.asr.segments_end_ts(res)
        t = self.commited[-1][1]
        if len(ends) > 1:
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2]+self.buffer_time_offset
            if e <= t:
                logger.debug(f"--- segment chunked at {e:2.2f}")
                self.chunk_at(e)
            else:
                logger.debug(f"--- last segment not within commited area")
        else:
            logger.debug(f"--- not enough segments to chunk")

    def chunk_at(self, time):
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds*self.SAMPLING_RATE):]
        self.buffer_time_offset = time

    def words_to_sentences(self, words):
        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        if self.tokenizer is None:
            # 如果 tokenizer 为空，而选择了 sentence trimming，会直接把所有词当成一个大句子
            return [(cwords[0][0], cwords[-1][1], t)]

        s = self.tokenizer.split(t)
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b,e,w = cwords.pop(0)
                w = w.strip()
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg,end,fsent))
                    break
                sent = sent[len(w):].strip()
        return out

    def finish(self):
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logger.debug(f"last, noncommited: {f}")
        self.buffer_time_offset += len(self.audio_buffer)/16000
        return f

    def to_flush(self, sents, sep=None, offset=0):
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b,e,t)


class VACOnlineASRProcessor(OnlineASRProcessor):
    '''Wraps OnlineASRProcessor with VAC (Voice Activity Controller).'''

    def __init__(self, online_chunk_size, *a, **kw):
        self.online_chunk_size = online_chunk_size
        self.online = OnlineASRProcessor(*a, **kw)
        import torch
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        from silero_vad_iterator import FixedVADIterator
        self.vac = FixedVADIterator(model)
        self.logfile = self.online.logfile
        self.init()

    def init(self):
        self.online.init()
        self.vac.reset_states()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        self.status = None
        self.audio_buffer = np.array([],dtype=np.float32)
        self.buffer_offset = 0

    def clear_buffer(self):
        self.buffer_offset += len(self.audio_buffer)
        self.audio_buffer = np.array([],dtype=np.float32)

    def insert_audio_chunk(self, audio):
        res = self.vac(audio)
        self.audio_buffer = np.append(self.audio_buffer, audio)

        if res is not None:
            frame = list(res.values())[0]-self.buffer_offset
            if 'start' in res and 'end' not in res:
                self.status = 'voice'
                send_audio = self.audio_buffer[frame:]
                self.online.init(offset=(frame+self.buffer_offset)/self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.clear_buffer()
            elif 'end' in res and 'start' not in res:
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[:frame]
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
            else:
                beg = res["start"]-self.buffer_offset
                end = res["end"]-self.buffer_offset
                self.status = 'nonvoice'
                send_audio = self.audio_buffer[beg:end]
                self.online.init(offset=(beg+self.buffer_offset)/self.SAMPLING_RATE)
                self.online.insert_audio_chunk(send_audio)
                self.current_online_chunk_buffer_size += len(send_audio)
                self.is_currently_final = True
                self.clear_buffer()
        else:
            if self.status == 'voice':
                self.online.insert_audio_chunk(self.audio_buffer)
                self.current_online_chunk_buffer_size += len(self.audio_buffer)
                self.clear_buffer()
            else:
                self.buffer_offset += max(0,len(self.audio_buffer)-self.SAMPLING_RATE)
                self.audio_buffer = self.audio_buffer[-self.SAMPLING_RATE:]

    def process_iter(self):
        if self.is_currently_final:
            return self.finish()
        elif self.current_online_chunk_buffer_size > self.SAMPLING_RATE*self.online_chunk_size:
            self.current_online_chunk_buffer_size = 0
            ret = self.online.process_iter()
            return ret
        else:
            print("no online update, only VAD", self.status, file=self.logfile)
            return (None, None, "")

    def finish(self):
        ret = self.online.finish()
        self.current_online_chunk_buffer_size = 0
        self.is_currently_final = False
        return ret


WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(",")

def create_tokenizer(lan):
    assert lan in WHISPER_LANG_CODES, "language must be Whisper's supported lang code."
    if lan == "uk":
        import tokenize_uk
        class UkrainianTokenizer:
            def split(self, text):
                return tokenize_uk.tokenize_sents(text)
        return UkrainianTokenizer()

    if lan in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh".split():
        from mosestokenizer import MosesTokenizer
        return MosesTokenizer(lan)

    if lan in "as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt".split():
        logger.debug(f"{lan} code is not supported by wtpsplit. Going to use None lang_code option.")
        lan = None

    from wtpsplit import WtP
    wtp = WtP("wtp-canine-s-12l-no-adapters")
    class WtPtok:
        def split(self, sent):
            return wtp.split(sent, lang_code=lan)
    return WtPtok()


def add_shared_args(parser):
    parser.add_argument('--min-chunk-size', type=float, default=1.0, help='Minimum audio chunk size in seconds.')
    parser.add_argument('--model', type=str, default='large-v2',
                        choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large-v3,large,large-v3-turbo".split(","),
                        help="Whisper model size (default: large-v2).")
    parser.add_argument('--model_cache_dir', type=str, default=None, help="Model cache dir.")
    parser.add_argument('--model_dir', type=str, default=None, help="Local model folder. Overrides --model.")
    parser.add_argument('--lan', '--language', type=str, default='auto', help="Language code or 'auto'.")
    parser.add_argument('--task', type=str, default='transcribe', choices=["transcribe","translate"],help="Transcribe or translate.")
    parser.add_argument('--backend', type=str, default="faster-whisper",
                        choices=["faster-whisper", "whisper_timestamped", "mlx-whisper", "openai-api"],
                        help='Backend to use for Whisper processing.')
    parser.add_argument('--vac', action="store_true", default=False, help='Use VAC = voice activity controller.')
    parser.add_argument('--vac-chunk-size', type=float, default=0.04, help='Chunk size in seconds for VAC.')
    parser.add_argument('--vad', action="store_true", default=False, help='Use VAD in the Whisper backend if supported.')
    parser.add_argument('--buffer_trimming', type=str, default="segment",
                        choices=["sentence", "segment"],
                        help='How to trim buffer: by completed sentence or by completed segment.')
    parser.add_argument('--buffer_trimming_sec', type=float, default=15,
                        help='Length threshold in seconds for trimming.')
    parser.add_argument("-l", "--log-level", dest="log_level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='DEBUG')


def asr_factory(args, logfile=sys.stderr):
    backend = args.backend
    if backend == "openai-api":
        logger.debug("Using OpenAI API.")
        asr = OpenaiApiASR(lan=args.lan)
    else:
        if backend == "faster-whisper":
            asr_cls = FasterWhisperASR
        elif backend == "mlx-whisper":
            asr_cls = MLXWhisper
        else:
            asr_cls = WhisperTimestampedASR
        size = args.model
        t = time.time()
        logger.info(f"Loading Whisper {size} model for {args.lan}...")
        asr = asr_cls(modelsize=size, lan=args.lan, cache_dir=args.model_cache_dir, model_dir=args.model_dir)
        e = time.time()
        logger.info(f"done. It took {round(e-t,2)} seconds.")

    if getattr(args, 'vad', False):
        logger.info("Setting VAD filter")
        asr.use_vad()
    if args.task == "translate":
        asr.set_translate_task()
        tgt_language = "en"
    else:
        tgt_language = args.lan

    if args.buffer_trimming == "sentence":
        tokenizer = create_tokenizer(tgt_language)
    else:
        tokenizer = None

    if args.vac:
        online = VACOnlineASRProcessor(args.min_chunk_size, asr, tokenizer, logfile=logfile,
                                       buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))
    else:
        online = OnlineASRProcessor(asr, tokenizer, logfile=logfile,
                                    buffer_trimming=(args.buffer_trimming, args.buffer_trimming_sec))
    return asr, online

def set_logging(args, logger, other="_server"):
    logging.basicConfig(format='%(levelname)s\t%(message)s')
    logger.setLevel(args.log_level)
    logging.getLogger("whisper_online"+other).setLevel(args.log_level)


# ---------------- 新增：可被外部脚本调用的函数 ----------------
def stt_streaming(
    audio_path: str,
    output_queue,
    start_time: float,
    min_chunk_size: float = 1.0,
    lan: str = 'auto',
    model: str = 'large-v2',
    model_cache_dir: str = None,
    model_dir: str = None,
    backend: str = 'faster-whisper',
    use_vad: bool = False,
    use_vac: bool = False,
    vac_chunk_size: float = 0.04,
    buffer_trimming: str = "segment",
    buffer_trimming_sec: float = 15.0,
    log_level: str = 'INFO'
):
    """
    供外部脚本调用：对指定音频文件进行“流式”ASR，并实时返回分段结果。

    Parameters
    ----------
    audio_path : str
        音频的绝对路径。要求是 16kHz 单声道(或可转换)。
    output_queue : multiprocessing.Queue
        用于输出流式 STT 结果的队列。
    其他参数略。

    Returns
    -------
    None
    """
    # 构造类似 argparse 的参数对象
    class Args: pass
    args = Args()
    args.min_chunk_size = min_chunk_size
    args.model = model
    args.model_cache_dir = model_cache_dir
    args.model_dir = model_dir
    args.lan = lan
    args.task = 'transcribe'
    args.backend = backend
    args.vac = use_vac
    args.vac_chunk_size = vac_chunk_size
    args.vad = use_vad
    args.buffer_trimming = buffer_trimming
    args.buffer_trimming_sec = buffer_trimming_sec
    args.log_level = log_level

    # 设置日志
    set_logging(args, logger)

    # 构造 asr 和 online 处理器
    asr, online = asr_factory(args, logfile=sys.stderr)

    # 先预加载音频以便获取总时长
    audio_data = load_audio(audio_path)
    
    SAMPLING_RATE = 16000
    duration = len(audio_data) / SAMPLING_RATE
    logger.info("Audio duration is: %2.2f seconds" % duration)

    # 模拟流式时, 先暖机一下
    warmup_chunk = audio_data[:16000] if len(audio_data) > 16000 else audio_data
    asr.transcribe(warmup_chunk)  # 预热一次

    # 这里使用 "comp_unaware" 模式去模拟流式分段: 按 min_chunk_size 循环处理
    beg = start_time
    chunk = min_chunk_size

    while True:
        end = beg + chunk
        if end > duration:
            end = duration
        chunk_data = audio_data[int(beg*SAMPLING_RATE): int(end*SAMPLING_RATE)]
        online.insert_audio_chunk(chunk_data)

        # 每插入一次块后，调用 process_iter() 获取最新的“流式”结果
        try:
            o = online.process_iter()
        except AssertionError as e:
            logger.error(f"assertion error: {repr(e)}")
            o = (None, None, "")
        if o and o[0] is not None:
            # 转换为 (emit_time_ms, start_ms, end_ms, text)
            # 这里为了模拟，我们令 emit_time ~ end
            emit_time_ms = end * 1000
            start_ms = o[0] * 1000
            end_ms = o[1] * 1000
            text = o[2]
            #print(f"IN FUNCTION emit_time_ms: {emit_time_ms}, start_ms: {start_ms}, end_ms: {end_ms}, text: {text}")
            # 将结果放入队列
            output_queue.put((emit_time_ms, start_ms, end_ms, text))

        if end >= duration:
            break
        beg = end

    # 音频块全部送入后，还要 finish() 一下，看看是否有最后残留输出
    last_o = online.finish()
    if last_o and last_o[0] is not None:
        # 统一再加一条
        emit_time_ms = duration * 1000
        start_ms = last_o[0] * 1000
        end_ms = last_o[1] * 1000
        text = last_o[2]
        output_queue.put((emit_time_ms, start_ms, end_ms, text))

    # 完成后，向队列发送一个特殊标志，表示结束
    output_queue.put(None)


# -------------------- 原来的命令行入口逻辑 --------------------
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path', type=str, help="Filename of 16kHz mono channel wav, on which live streaming is simulated.")
    add_shared_args(parser)
    parser.add_argument('--start_at', type=float, default=0.0, help='Start processing audio at this time.')
    parser.add_argument('--offline', action="store_true", default=False, help='Offline mode.')
    parser.add_argument('--comp_unaware', action="store_true", default=False, help='Computationally unaware simulation.')
    args = parser.parse_args()

    logfile = sys.stderr

    if args.offline and args.comp_unaware:
        logger.error("Cannot enable both --offline and --comp_unaware. Exiting.")
        sys.exit(1)

    set_logging(args, logger)
    audio_path = args.audio_path

    SAMPLING_RATE = 16000
    duration = len(load_audio(audio_path)) / SAMPLING_RATE
    logger.info("Audio duration is: %2.2f seconds" % duration)

    asr, online = asr_factory(args, logfile=logfile)

    if args.vac:
        min_chunk = args.vac_chunk_size
    else:
        min_chunk = args.min_chunk_size

    # 预热
    a = load_audio_chunk(audio_path,0,1)
    asr.transcribe(a)

    beg = args.start_at
    start = time.time() - beg

    def output_transcript(o, now=None):
        if now is None:
            now = time.time()-start
        if o[0] is not None:
            print("%1.4f %1.0f %1.0f %s" % (now*1000, o[0]*1000, o[1]*1000, o[2]), flush=True)
        else:
            pass

    if args.offline:
        a = load_audio(audio_path)
        online.insert_audio_chunk(a)
        try:
            o = online.process_iter()
        except AssertionError as e:
            logger.error(f"assertion error: {repr(e)}")
            o = (None, None, "")
        output_transcript(o)
        now = None

    elif args.comp_unaware:
        end = beg + min_chunk
        while True:
            a = load_audio_chunk(audio_path, beg, end)
            online.insert_audio_chunk(a)
            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {repr(e)}")
                o = (None, None, "")
            output_transcript(o, now=end)
            logger.debug(f"## last processed {end:.2f}s")

            if end >= duration:
                break
            beg = end
            if end + min_chunk > duration:
                end = duration
            else:
                end += min_chunk
        now = duration

    else:
        end = 0
        while True:
            now = time.time() - start
            if now < end + min_chunk:
                time.sleep(min_chunk + end - now)
            end = time.time() - start
            a = load_audio_chunk(audio_path,beg,end)
            beg = end
            online.insert_audio_chunk(a)

            try:
                o = online.process_iter()
            except AssertionError as e:
                logger.error(f"assertion error: {e}")
                o = (None, None, "")
            output_transcript(o)
            now = time.time() - start
            logger.debug(f"## last processed {end:.2f} s, now is {now:.2f}, latency={now-end:.2f}")

            if end >= duration:
                break
        now = None

    o = online.finish()
    output_transcript(o, now=now)
