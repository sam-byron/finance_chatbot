# finance_chatbot

python3 prepare_data_mp.py --config_path model_open_web_full.json
python3 prepare_data_mp.py --config_path model_open_web_full.json --sanitize

python3 iter_data_loader.py --config_path model_open_web_full.json

accelerate launch chatbotOWTAcc_mp.py --config_path model_open_web_full.json

lm-eval --model hf \
        --model_args "pretrained=./model_open_web_full/checkpoint.pt,tokenizer=gpt2" \
        --tasks wikitext \
        --batch_size 8 \
        --device cuda

python3 inter_chat_acc.py  --config_path model_open_web_full.json --model_path model_open_web_full/checkpoint.pt