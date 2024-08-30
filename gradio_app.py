import os
import sys
import gradio as gr
import random
from viewcrafter import ViewCrafter
from configs.infer_config import get_parser

# i2v_examples_1024 = [
#     ['prompts/1024/astronaut04.png', 'a man in an astronaut suit playing a guitar', 50, 7.5, 1.0, 6, 123],
#     ['prompts/1024/bloom01.png', 'time-lapse of a blooming flower with leaves and a stem', 50, 7.5, 1.0, 10, 123],
#     ['prompts/1024/girl07.png', 'a beautiful woman with long hair and a dress blowing in the wind', 50, 7.5, 1.0, 10, 123],
#     ['prompts/1024/pour_bear.png', 'pouring beer into a glass of ice and beer', 50, 7.5, 1.0, 10, 123],
#     ['prompts/1024/robot01.png', 'a robot is walking through a destroyed city', 50, 7.5, 1.0, 10, 123],
#     ['prompts/1024/firework03.png', 'fireworks display', 50, 7.5, 1.0, 10, 123],
# ]

max_seed = 2 ** 31


def viewcrafter_demo(opts):
    css = """#input_img {max-width: 1024px !important} #output_vid {max-width: 1024px; max-height:576px} #random_button {max-width: 100px !important}"""
    image2video = ViewCrafter(opts, gradio = True)
    with gr.Blocks(analytics_enabled=False, css=css) as viewcrafter_iface:
        gr.Markdown("<div align='center'> <h1> ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis </span> </h1> \
                      <h2 style='font-weight: 450; font-size: 1rem; margin: 0rem'>\
                        <a href=''>Wangbo Yu</a>, \
                        <a href=''>Jinbo Xing</a>, <a href=''>Li Yuan</a>, \
                        <a href=''>Wenbo Hu</a>, <a href=''>Xiaoyu Li</a>,\
                        <a href=''>Zhipeng Huang</a>, <a href=''>Xiangjun Gao</a>,\
                        <a href=''>Tien-Tsin Wong</a>,\
                        <a href=''>Ying Shan</a>\
                        <a href=''>Yonghong Tian</a>\
                    </h2> \
                     <a style='font-size:18px;color: #FF5DB0' href='https://github.com/Doubiiu/viewcrafter'> [Guideline] </a>\
                     <a style='font-size:18px;color: #000000' href='https://arxiv.org/abs/2310.12190'> [ArXiv] </a>\
                     <a style='font-size:18px;color: #000000' href='https://doubiiu.github.io/projects/viewcrafter/'> [Project Page] </a>\
                     <a style='font-size:18px;color: #000000' href='https://github.com/Doubiiu/viewcrafter'> [Github] </a> </div>") 
                
        #######image2video######
        with gr.Tab(label="ViewCrafter_25, 'single_view_txt' mode"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            i2v_input_image = gr.Image(label="Input Image",elem_id="input_img")
                        with gr.Row():
                            i2v_elevation = gr.Text(label='elevation')
                        with gr.Row():
                            i2v_d_phi = gr.Text(label='d_phi sequence, should start with 0')
                        with gr.Row():
                            i2v_d_theta = gr.Text(label='d_theta sequence, should start with 0')
                        with gr.Row():
                            i2v_d_r = gr.Text(label='d_r sequence, should start with 0')
                        with gr.Row():
                            i2v_center_scale = gr.Slider(minimum=0.1, maximum=2, step=0.1, elem_id="i2v_center_scale", label="center_scale", value=1)
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
                            i2v_render_video = gr.Video(label="Point Cloud Render",elem_id="render_vid",autoplay=True,show_share_button=True)
                        with gr.Row():
                            i2v_output_video = gr.Video(label="Generated Video",elem_id="output_vid",autoplay=True,show_share_button=True)

                # gr.Examples(examples=i2v_examples_1024,
                #             inputs=[i2v_input_image, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_eta, i2v_motion, i2v_seed],
                #             outputs=[i2v_output_video],
                #             fn = image2video.get_image,
                #             cache_examples=False,
                # )

            # image2video.run_gradio(i2v_input_image='test/images/boy.png', i2v_elevation='10', i2v_d_phi='0 40', i2v_d_theta='0 0', i2v_d_r='0 0', i2v_center_scale=1, i2v_steps=50, i2v_seed=123)
            i2v_end_btn.click(inputs=[i2v_input_image, i2v_elevation, i2v_d_phi, i2v_d_theta, i2v_d_r, i2v_center_scale, i2v_steps, i2v_seed],
                            outputs=[i2v_traj_video,i2v_render_video,i2v_output_video],
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
    viewcrafter_iface.launch(server_name='11.204.23.92', server_port=80, max_threads=1,debug=True)
