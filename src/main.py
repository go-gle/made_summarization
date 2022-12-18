from predictors import PGenPredictor, MBartPredictor, ExtractionPGenPredictor

import gradio as gr
import torch


PG_MODEL_PATH = './pointer_gazeta.pth'
PG_VOCAB_PATH = './gazeta_voc.pth'
EXTR_MODEL_PATH = './extractor.pth'

def apply_models(text):
    mbart_abs = mbart.predict_one_sample(text)
    pg_abs = pg.predict_one_sample(text)
    botup_abs = botup.predict_one_sample(text)
    return mbart_abs, pg_abs, botup_abs


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device is {device}')
    print('Loading Pointer Gen')
    pg = PGenPredictor(
        model_path=PG_MODEL_PATH,
        vocab_path=PG_VOCAB_PATH,
        device=device)
    print('Loading MBART')
    mbart = MBartPredictor(device=device)
    print('Loading Extractor')
    botup = ExtractionPGenPredictor(
        ext_model_path=EXTR_MODEL_PATH,
        pg_model_path=PG_MODEL_PATH,
        pg_vocab_path=PG_VOCAB_PATH,
        device=device,
        threshold=0.5)
   
    with gr.Blocks() as demo:
        btn = gr.Button("Summarize")
        inp = gr.Textbox(lines=30, placeholder="Put News Here")
        mbart_out = gr.Textbox(label="Mbart")
        pgen_out = gr.Textbox(label="Pointer Gen")
        botup_out = gr.Textbox(label="Bottom-up Summarization")
        btn.click(fn=apply_models, inputs=inp, outputs=[mbart_out, pgen_out, botup_out])
    print('Launch Gradio')
    demo.launch(server_name="0.0.0.0", server_port=8889)

