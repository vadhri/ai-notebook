This repository is a list of assorted information on AI and topics below.

|Topic|Comments|
|-|-|
|Audio|ASR|Basic voice dataset handling and visualization ( MEL spectrogram) Standard HF pipeline to <ul><li>load the fb-voxpopuli dataset to extract data sample and do asr.<li>ASR and text-speech basic examples</ul><p>[basics](./audio/basics/)|
||Model fine tuning with seq2seq and whisper tiny model [asr_model_finetuning.ipynb](./audio/asr_model_finetuning.ipynb)|
||Music genre classification using gztan dataset [music_genre_classification_transformers.ipynb](./audio/music_genre_classification_transformers.ipynb)|
||Fine-tune Speech5 TTS model using voxpopuli dataset. [text_to_speech.ipynb](./audio/text_to_speech.ipynb)|
||Speech to speech HF space [speech-to-speech.py](./audio/speech-to-speech.py) 
|Images|Basics downsampling, transpose.|
||Usage of noise in diffusers <li>linear and cosine scheduling [add_noise.ipynb](./images/Noise%20Samples/add_noise.ipynb) <li> Noise prediction [predict_noise](./images/minst/PredictNoise.ipynb)|
||Modified UNet for diffusers (example) [Modified_UNet.ipynb](./images/minst/Modified_UNet.ipynb)|
||Unet with Fashion_minst [diffusion_fashion_minst.ipynb](./images/minst/diffusion_fashion_minst.ipynb)|
||Adding time to noise  [Time.ipynb](./images/minst/Time+Noise+UNet.ipynb)|
|Text|LLM pre-req <li>Basic tokenization and pipelines [basics](./text/nlp-hf/)<li> Self attn block math [selfattn.ipynb](./text/decoder_only/self-attn-maths.ipynb)<li>Bigram Langauge Model [BLM.ipynb](./text/decoder_only/bigram_language_model.ipynb)|
||Nanogpt implementation of multihead attn [nano.ipynb](./text/decoder_only/nano_gpt.ipynb)|
||LLM Fine tuning based on bert-base-uncased <li> Training with HF  [sequence_classification.ipynb](./text/fine-tuning-llm/sequence_classification.ipynb)<li>Training with Pytorch [sequence_classification_pytorch.ipynb](./text/fine-tuning-llm/sequence_classification_pytorch.ipynb)|
||Chatbots <li> Vectordb similarity search [vector](./text/langchain/vectordb/)<li>PDF chat based on llama 2 and 3 [pdf llama](./text/langchain/pdf_chat-llama/)<li> Finetune fb_opt-125m with chat corpus bittext [LLM_FinTune_fb_opt_125m.ipynb](./text/langchain/LLM_FineTuning.ipynb)<li> Finetune fb_opt-350m with chat corpus bittext [LLM_FinTune_fb_opt_350m.ipynb](./text/langchain/LLM_FineTuning_fb_opt_350mn.ipynb)|
||question-answer finetuning [qa](./text/fine-tuning-llm/question_answering/)|
||Multi-Lora inference [inference.ipynb](./text/inference/llm_serving.ipynb)|
|Deep Reinforcement Learning|The 8 units in deeplearning belong to the [Deep RL course](https://huggingface.co/learn/deep-rl-course/en/unit0/introduction)<li>LunarLander-v2 [Unit1](./deep-rl/Unit1/)<li>Taxi-v3 and Frozenlake-v1 [Unit2](./deep-rl/Unit2/)<li>SpacesInvadersNoFrameskip-v4 [Unit3](./deep-rl/Unit3/)<li>CartPole-v1 and PixelCopter[Unit4](./deep-rl/Unit4/)<li>Pyramids and SnowballTarget[ Unit5.ipynb](./deep-rl/Unit5.ipynb)  <li>PandaReachDense-v3 [Unit6.ipynb](./deep-rl/Unit6.ipynb) <li>UnityML SoccerTwos [Unit7.ipynb](./deep-rl/Unit7.ipynb) <li> PPO cartpole-v1 [Unit8](./deep-rl/Unit8/)|
|UDL book|Solutions for [udl.book](https://github.com/udlbook/udlbook/tree/main/Notebooks)|