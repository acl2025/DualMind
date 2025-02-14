# DualMind: A Fast-Slow Thinking Agent Framework for Meeting Assistance
![DualMind Architecture](./figures/会议系统agent1230.drawio.jpg)
DualMind is a novel dual-process meeting assistance system designed to balance rapid reaction and careful reasoning in real-time meeting scenarios. The system features a fast-thinking Talker agent for routine queries and a slow-thinking Planner agent for complex tasks, along with comprehensive datasets and evaluation frameworks.

## Key Features

- **Dual-Process Architecture**: Fast-thinking Talker (sub-900ms latency) and slow-thinking Planner for optimal handling of varying query complexities
- **AISHELL-Agent Dataset**: Enhanced meeting assistance dataset with voice-cloned interactions
- **AMBER Benchmark Framework**: Multi-criteria evaluation suite adapted from CompassJudger and Prometheus
- **Real-time Processing**: Efficient audio detection and streaming STT capabilities
- **Parallel GPU Scheduling**: Optimized resource utilization across multiple GPUs

## System Requirements

- Python 3.11.4
- PyTorch 2.1.0
- 2× NVIDIA RTX 3090 GPUs
- CUDA 11.8+

## Installation

```bash
git clone https://github.com/yourusername/dualmind.git
cd Dualmind
pip install -r requirements.txt
```

## Quick Start

1. Start the DualMind audio streaming:
```bash
python test_agent_audio_streaming.py.py
```

2. Run real-time meeting assistance:
```bash
python test_agent_audio_streaming_only_latency.py
```

2. Run experiment:
```bash
python test_agent_audio_experiement_audio_segment_only.py
```

## Architecture Overview

DualMind consists of four main components:

1. **Input Processing Layer**
   - Audio-based keyword spotting
   - Whisper-based STT module
   - Query complexity assessment

2. **Dual-Agent Intelligence Layer**
   - Talker: Fast response generation (Qwen2-Audio)
   - Planner: Complex reasoning (Qwen2.5-14B)

3. **Tool Integration Layer**
   - RAG-based knowledge retrieval
   - Meeting context management

4. **Output Management Layer**
   - Response generation
   - TTS & Avatar modules

## Performance

- **Response Time**: 1500ms reduction for routine queries
- **Quality Improvement**: 22.5% better outcomes on complex tasks
- **Latency Metrics**:
  - STT processing: 53ms per token
  - Talker first token: 210ms
  - Planner first token: 520ms

## Dataset

The AISHELL-Agent dataset includes:
- 208 enhanced meeting recordings
- Multi-speaker conversations
- Voice-cloned agent responses
- Diverse query complexities

## Evaluation

The AMBER framework provides:
- CompassJudger scoring (1-10 scale)
- Prometheus evaluation (1-5 scale)
- Latency measurements
- Human evaluation metrics

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Citation

If you use DualMind in your research, please cite:

```bibtex
@inproceedings{dualmind2024,
  title={DualMind: A Fast-Slow Thinking Agent Real-Time Framework for Meeting Assistance},
  author={Your Name},
  booktitle={Proceedings of ACL 2024},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- AISHELL corpus team
- CompassJudger and Prometheus framework developers
- OpenAI Whisper team
