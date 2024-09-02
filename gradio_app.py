import os
import sys
import gradio as gr
import random
from viewcrafter import ViewCrafter
from configs.infer_config import get_parser

i2v_examples = [
    ['test/images/boy.png', 0, 1.0, '0 40', '0 0', '0 0',  50, 123],
    ['test/images/car.jpg', 0, 1.0, '0 -35', '0 0', '0 -0.1',  50, 123],
    ['test/images/fruit.png', 0, 1.0, '0 -3 -15 -20 -17 -5 0', '0 -2 -5 -10 -8 -5 0 2 5 3 0', '0 0',  50, 123],
    ['test/images/room.png', 10, 1.0, '0 3 10 20 17 10 0', '0 -2 -8 -6 0 2 5 3 0', '0 -0.02 -0.09 -0.16 -0.09 0',  50, 123],
    ['test/images/castle.png', 0, 1.0, '0 30', '0 -1 -5 -4 0 1 5 4 0', '0 -0.2',  50, 123],
]

max_seed = 2 ** 31


def viewcrafter_demo(opts):
    css = """#input_img {max-width: 1024px !important} #output_vid {max-width: 1024px; max-height:576px} #random_button {max-width: 100px !important}"""
    image2video = ViewCrafter(opts, gradio = True)
    with gr.Blocks(analytics_enabled=False, css=css) as viewcrafter_iface:
        gr.Markdown("<div align='center'> <h1> ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis </span> </h1> \
                      <h2 style='font-weight: 450; font-size: 1rem; margin: 0rem'>\
                        <a href='https://scholar.google.com/citations?user=UOE8-qsAAAAJ&hl=zh-CN'>Wangbo Yu</a>, \
                        <a href='https://doubiiu.github.io/'>Jinbo Xing</a>, <a href=''>Li Yuan</a>, \
                        <a href='https://wbhu.github.io/'>Wenbo Hu</a>, <a href='https://xiaoyu258.github.io/'>Xiaoyu Li</a>,\
                        <a href=''>Zhipeng Huang</a>, <a href='https://scholar.google.com/citations?user=qgdesEcAAAAJ&hl=en/'>Xiangjun Gao</a>,\
                        <a href='https://www.cse.cuhk.edu.hk/~ttwong/myself.html/'>Tien-Tsin Wong</a>,\
                        <a href='https://scholar.google.com/citations?hl=en&user=4oXBp9UAAAAJ&view_op=list_works&sortby=pubdate/'>Ying Shan</a>\
                        <a href=''>Yonghong Tian</a>\
                    </h2> \
                     <a style='font-size:18px;color: #FF5DB0' href='https://github.com/Drexubery/ViewCrafter/blob/main/docs/render_help.md'> [Guideline] </a>\
                     <a style='font-size:18px;color: #000000' href=''> [ArXiv] </a>\
                     <a style='font-size:18px;color: #000000' href='https://drexubery.github.io/ViewCrafter/'> [Project Page] </a>\
                     <a style='font-size:18px;color: #000000' href='https://github.com/Drexubery/ViewCrafter'> [Github] </a> </div>") 
                
        #######image2video######
        with gr.Tab(label="ViewCrafter_25, 'single_view_txt' mode"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            i2v_input_image = gr.Image(label="Input Image",elem_id="input_img")
                        with gr.Row():
                            i2v_elevation = gr.Slider(minimum=-45, maximum=45, step=1, elem_id="elevation", label="elevation", value=5)
                        with gr.Row():
                            i2v_center_scale = gr.Slider(minimum=0.1, maximum=2, step=0.1, elem_id="i2v_center_scale", label="center_scale", value=1)
                        with gr.Row():
                            i2v_d_phi = gr.Text(label='d_phi sequence, should start with 0')
                        with gr.Row():
                            i2v_d_theta = gr.Text(label='d_theta sequence, should start with 0')
                        with gr.Row():
                            i2v_d_r = gr.Text(label='d_r sequence, should start with 0')
                        with gr.Row():
                            i2v_steps = gr.Slider(minimum=1, maximum=50, step=1, elem_id="i2v_steps", label="Sampling steps", value=50)
                        with gr.Row():
                            i2v_seed = gr.Slider(label='Random Seed', minimum=0, maximum=max_seed, step=1, value=123)
                        i2v_end_btn = gr.Button("Generate")
                    # with gr.Tab(label='Result'):
                    with gr.Column():
                        with gr.Row():
                            i2v_traj_video = gr.Video(label="Camera Trajectory",elem_id="traj_vid",autoplay=True,show_share_button=True)
                        with gr.Row():
                            i2v_output_video = gr.Video(label="Generated Video",elem_id="output_vid",autoplay=True,show_share_button=True)

                gr.Examples(examples=i2v_examples,
                            inputs=[i2v_input_image, i2v_elevation, i2v_center_scale, i2v_d_phi, i2v_d_theta, i2v_d_r, i2v_steps, i2v_seed],
                            outputs=[i2v_traj_video,i2v_output_video],
                            fn = image2video.run_gradio,
                            cache_examples=False,
                )

            # image2video.run_gradio(i2v_input_image='test/images/boy.png', i2v_elevation='10', i2v_d_phi='0 40', i2v_d_theta='0 0', i2v_d_r='0 0', i2v_center_scale=1, i2v_steps=50, i2v_seed=123)
            i2v_end_btn.click(inputs=[i2v_input_image, i2v_elevation, i2v_center_scale, i2v_d_phi, i2v_d_theta, i2v_d_r, i2v_steps, i2v_seed],
                            outputs=[i2v_traj_video,i2v_output_video],
                            fn = image2video.run_gradio
            )

    return viewcrafter_iface


if __name__ == "__main__":
    parser = get_parser() # infer_config.py
    opts = parser.parse_args() # default device: 'cuda:0'
    opts.save_dir = './gradio_tmp'
    os.makedirs(opts.save_dir,exist_ok=True)
    viewcrafter_iface = viewcrafter_demo(opts)
    viewcrafter_iface.queue(max_size=1)
    # viewcrafter_iface.launch(max_threads=1)
    viewcrafter_iface.launch(server_name='127.0.0.1', server_port=80, max_threads=1,debug=False)
