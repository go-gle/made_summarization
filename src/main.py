from src.predictors import PGenPredictor, MBartPredictor
import gradio as gr


def apply_models(text):
    mbart_abs = mbart.predict_one_sample(text)
    pg_abs = pg.predict_one_sample(text)
    return mbart_abs, pg_abs


if __name__ == '__main__':
    print('Loading Pointer Gen')
    pg = PGenPredictor(
    model_path='/Users/ruagcg2/Downloads/pointer_gazeta_voc33_rouge30.pth',
    vocab_path='/Users/ruagcg2/Downloads/gazeta_voc.pth')
    print('Loading MBART')
    mbart = MBartPredictor()
   
    with gr.Blocks() as demo:
        btn = gr.Button("Summarize")
        inp = gr.Textbox(lines=30, placeholder="Put News Here")
        mbart_out = gr.Textbox(label="Mbart")
        pgen_out = gr.Textbox(label="Pointer Gen")
        btn.click(fn=apply_models, inputs=inp, outputs=[mbart_out, pgen_out])
    print('Launch Gradio')
    demo.launch()

