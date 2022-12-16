from predictors import PGenPredictor, MBartPredictor

import gradio as gr
import torch


def apply_models(text):
    mbart_abs = mbart.predict_one_sample(text)
    pg_abs = pg.predict_one_sample(text)
    return mbart_abs, pg_abs

def dummy_output(text):
    return 'first', 'second'

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device is {device}')
    #print('Loading Pointer Gen')
    #pg = PGenPredictor(
    #model_path='./pointer_gazeta.pth',
    #vocab_path='./gazeta_voc.pth',
    #device=device)
    #print('Loading MBART')
    #mbart = MBartPredictor(device=device)
   
    with gr.Blocks() as demo:
        btn = gr.Button("Summarize")
        inp = gr.Textbox(lines=30, placeholder="Put News Here")
        mbart_out = gr.Textbox(label="Mbart")
        pgen_out = gr.Textbox(label="Pointer Gen")
        btn.click(fn=dummy_output, inputs=inp, outputs=[mbart_out, pgen_out])
    print('Launch Gradio')
    demo.launch(server_name="0.0.0.0", server_port=8889)

