# 定义全局变量用于模型的共享
global_kws_models = None


def init_kws_worker():
    global global_kws_models
    print("Initializing KWS models...")
    # 在子进程中导入，避免主进程初始化 CUDA
    from wewks.wekws.wekws.bin.stream_kws_ctc import KeyWordSpotter
    # 初始化关键词检测模型
    # 检测词1
    kws_model_1 = KeyWordSpotter(
        ckpt_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/avg_30.pt',
        config_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/config.yaml',
        token_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/tokens.txt',
        lexicon_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/lexicon.txt',
        threshold=0.0001,
        min_frames=3,
        max_frames=3000,
        interval_frames=30,
        score_beam=10,
        path_beam=40,
        gpu=-1,
        is_jit_model=False,
    )
    kws_model_1.set_keywords("好交交")

    # 检测词2
    kws_model_2 = KeyWordSpotter(
        ckpt_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/avg_30.pt',
        config_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/config.yaml',
        token_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/tokens.txt',
        lexicon_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/lexicon.txt',
        threshold=0.0001,
        min_frames=3,
        max_frames=3000,
        interval_frames=30,
        score_beam=10,
        path_beam=40,
        gpu=-1,
        is_jit_model=False,
    )
    kws_model_2.set_keywords("好教教")

    # 检测词3
    kws_model_3 = KeyWordSpotter(
        ckpt_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/avg_30.pt',
        config_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/config.yaml',
        token_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/tokens.txt',
        lexicon_path='/home/leon/agent/wewks/wekws/examples/hi_xiaowen/s0/kws_wenwen_dstcn_ctc/lexicon.txt',
        threshold=0.0001,
        min_frames=3,
        max_frames=3000,
        interval_frames=30,
        score_beam=10,
        path_beam=40,
        gpu=-1,
        is_jit_model=False,
    )
    kws_model_3.set_keywords("好焦焦")

    global_kws_models = (kws_model_1, kws_model_2, kws_model_3)

def kws_worker(chunk_wav):
    global global_kws_models
    kws_model_1, kws_model_2, kws_model_3 = global_kws_models
    # 重置模型状态
    kws_model_1.reset_all()
    kws_model_2.reset_all()
    kws_model_3.reset_all()
    # 进行关键词检测
    kws_result_1 = kws_model_1.forward(chunk_wav)
    kws_result_2 = kws_model_2.forward(chunk_wav)
    kws_result_3 = kws_model_3.forward(chunk_wav)
    return kws_result_1, kws_result_2, kws_result_3