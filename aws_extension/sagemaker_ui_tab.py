import logging
import os

import gradio as gr
import requests

import utils
from aws_extension import sagemaker_ui
from dreambooth_on_cloud.train import get_sorted_cloud_dataset
from modules.ui_common import create_refresh_button
from modules.ui_components import FormRow
import modules.ui
from utils import get_variable_from_json, save_variable_to_json

logger = logging.getLogger(__name__)
logger.setLevel(utils.LOGGING_LEVEL)

async_inference_choices = ["ml.g4dn.2xlarge", "ml.g4dn.4xlarge", "ml.g4dn.8xlarge", "ml.g4dn.12xlarge", "ml.g5.2xlarge",
                           "ml.g5.4xlarge", "ml.g5.8xlarge", "ml.g5.12xlarge", "ml.g5.12xlarge"]


test_connection_result = None
api_gateway_url = None
api_key = None


def on_ui_tabs():
    buildin_model_list = ['AWS JumpStart Model', 'AWS BedRock Model', 'Hugging Face Model']
    with gr.Blocks() as sagemaker_interface:
        with gr.Tab(label='API and User Settings'):
            with gr.Row():
                with gr.Column(variant="panel", scale=1):
                    api_tab = api_setting_tab()
                with gr.Column(variant="panel", scale=2):
                    user_settings_tab()
        with gr.Tab(label='Cloud Assets Management', variant='panel'):
            with gr.Row():
                # todo: the output message is not right yet
                model_upload = model_upload_tab()
                sagemaker_part = sagemaker_endpoint_tab()
            with gr.Row(visible=False):
                with gr.Row(equal_height=True, elem_id="aws_sagemaker_ui_row", visible=False):
                    sm_load_params = gr.Button(value="Load Settings", elem_id="aws_load_params", visible=False)
                    sm_save_params = gr.Button(value="Save Settings", elem_id="aws_save_params", visible=False)
                    sm_train_model = gr.Button(value="Train", variant="primary", elem_id="aws_train_model", visible=False)
                    sm_generate_checkpoint = gr.Button(value="Generate Ckpt", elem_id="aws_gen_ckpt", visible=False)

        with gr.Tab(label='Create AWS dataset', variant='panel'):
            dataset_tab()

    return (sagemaker_interface, "Amazon SageMaker", "sagemaker_interface"),


def api_setting_tab():
    with gr.Blocks() as api_setting:
        gr.HTML(value="<u><b>AWS Connection Setting</b></u>")
        gr.HTML(value="Enter your API URL & Token to start the connection.")
        global api_gateway_url
        api_gateway_url = get_variable_from_json('api_gateway_url')
        global api_key
        api_key = get_variable_from_json('api_token')
        with gr.Row():
            api_url_textbox = gr.Textbox(value=api_gateway_url, lines=1,
                                         placeholder="Please enter API Url of Middle", label="API Url",
                                         elem_id="aws_middleware_api")

            def update_api_gateway_url():
                global api_gateway_url
                api_gateway_url = get_variable_from_json('api_gateway_url')
                return api_gateway_url

            # modules.ui.create_refresh_button(api_url_textbox,
            # get_variable_from_json('api_gateway_url'),
            # lambda: {"value": get_variable_from_json('api_gateway_url')}, "refresh_api_gate_way")
            modules.ui.create_refresh_button(api_url_textbox, update_api_gateway_url,
                                             lambda: {"value": api_gateway_url}, "refresh_api_gateway_url")
        with gr.Row():
            def update_api_key():
                global api_key
                api_key = get_variable_from_json('api_token')
                return api_key

            api_token_textbox = gr.Textbox(value=api_key, lines=1, placeholder="Please enter API Token",
                                           label="API Token", elem_id="aws_middleware_token")
            modules.ui.create_refresh_button(api_token_textbox, update_api_key, lambda: {"value": api_key},
                                             "refresh_api_token")

        global test_connection_result
        test_connection_result = gr.Label(title="Output")
        aws_connect_button = gr.Button(value="Update Setting", variant='primary', elem_id="aws_config_save")
        aws_connect_button.click(_js="update_auth_settings",
                                 fn=update_connect_config,
                                 inputs=[api_url_textbox, api_token_textbox],
                                 outputs=[test_connection_result])
        aws_test_button = gr.Button(value="Test Connection", variant='primary', elem_id="aws_config_test")
        aws_test_button.click(test_aws_connect_config, inputs=[api_url_textbox, api_token_textbox],
                              outputs=[test_connection_result])

        with gr.Row():
            with gr.Accordion("Disclaimer", open=False):
                gr.HTML(
                    value="""You should perform your own independent assessment, and take measures to ensure 
                                that you comply with your own specific quality control practices and standards, and the 
                                local rules, laws, regulations, licenses and terms of use that apply to you, your content, 
                                and the third-party generative AI service in this web UI. Amazon Web Services has no control
                                 or authority over the third-party generative AI service in this web UI, and does not make
                                  any representations or warranties that the third-party generative AI service is secure,
                                   virus-free, operational, or compatible with your production environment and standards.""")
    return api_setting


def user_settings_tab():
    gr.HTML(value="<u><b>Manage User's Access</b></u>")
    with gr.Row(variant='panel') as user_tab:
        with gr.Column(scale=1):
            def roles():
                # todo: fixme
                return ['IT Operator', 'Designer']

            gr.HTML(value="<b>Update a User Setting</b>")
            username = gr.Textbox(placeholder="Please enter Enter a username", label="User name")
            pwd = gr.Textbox(placeholder="Please enter Enter password", label="Password", type='password')
            user_roles = gr.Dropdown(choices=roles(), multiselect=True, label="User Role")
            upsert_user_button = gr.Button(value="Upsert a User", variant='primary')
            disable_user_button = gr.Button(value="Disable a User", variant='primary')
        with gr.Column(scale=2):
            def list_users():
                # todo: fixme
                return [
                    ['cyanda', 'IT Operator, Designer', 'cyanda'],
                    ['alvindaiyan', 'Designer', 'cyanda']
                ]

            gr.HTML(value="<b>Users Table</b>")
            user_table = gr.Dataframe(
                headers=["name", "role", "created by"],
                datatype=["str", "str", "str"],
                max_rows=10,
                value=list_users
            )

            def choose_user(evt: gr.SelectData):
                if evt.index[1] != 0:
                    return gr.skip(), gr.skip(), gr.skip()

                # todo: to be done
                return 'cyanda', '123123', ['IT Operator', 'Designer']

            user_table.select(fn=choose_user, inputs=[], outputs=[username, pwd, user_roles])

            with gr.Row():
                current_page_token = gr.State({
                    'previous_token': '',
                    'next_token': '',
                })
                next_page = gr.Button(value="Next Page", variant='primary')
                previous_page = gr.Button(value="Previous Page", variant='primary')

    return user_tab


def model_upload_tab():
    with gr.Column() as upload_tab:
        gr.HTML(value="<b>Upload Model to Cloud</b>")
        # sagemaker_html_log = gr.HTML(elem_id=f'html_log_sagemaker')
        with gr.Column(variant="panel"):
            gr.HTML(value="<b>Upload Model to S3 from WebUI</b>")
            gr.HTML(value="Refresh to select the model to upload to S3")
            exts = (".bin", ".pt", ".pth", ".safetensors", ".ckpt")
            root_path = os.getcwd()
            model_folders = {
                "ckpt": os.path.join(root_path, "models", "Stable-diffusion"),
                "text": os.path.join(root_path, "embeddings"),
                "lora": os.path.join(root_path, "models", "Lora"),
                "control": os.path.join(root_path, "models", "ControlNet"),
                "hyper": os.path.join(root_path, "models", "hypernetworks"),
                "vae": os.path.join(root_path, "models", "VAE"),
            }

            def scan_sd_ckpt():
                model_files = os.listdir(model_folders["ckpt"])
                # filter non-model files not in exts
                model_files = [f for f in model_files if os.path.splitext(f)[1] in exts]
                model_files = [os.path.join(model_folders["ckpt"], f) for f in model_files]
                return model_files

            def scan_textual_inversion_model():
                model_files = os.listdir(model_folders["text"])
                # filter non-model files not in exts
                model_files = [f for f in model_files if os.path.splitext(f)[1] in exts]
                model_files = [os.path.join(model_folders["text"], f) for f in model_files]
                return model_files

            def scan_lora_model():
                model_files = os.listdir(model_folders["lora"])
                # filter non-model files not in exts
                model_files = [f for f in model_files if os.path.splitext(f)[1] in exts]
                model_files = [os.path.join(model_folders["lora"], f) for f in model_files]
                return model_files

            def scan_control_model():
                model_files = os.listdir(model_folders["control"])
                # filter non-model files not in exts
                model_files = [f for f in model_files if os.path.splitext(f)[1] in exts]
                model_files = [os.path.join(model_folders["control"], f) for f in model_files]
                return model_files

            def scan_hypernetwork_model():
                model_files = os.listdir(model_folders["hyper"])
                # filter non-model files not in exts
                model_files = [f for f in model_files if os.path.splitext(f)[1] in exts]
                model_files = [os.path.join(model_folders["hyper"], f) for f in model_files]
                return model_files

            def scan_vae_model():
                model_files = os.listdir(model_folders["vae"])
                # filter non-model files not in exts
                model_files = [f for f in model_files if os.path.splitext(f)[1] in exts]
                model_files = [os.path.join(model_folders["vae"], f) for f in model_files]
                return model_files

            with FormRow(elem_id="model_upload_form_row_01"):
                sd_checkpoints_path = gr.Dropdown(label="SD Checkpoints", choices=sorted(scan_sd_ckpt()),
                                                  elem_id="sd_ckpt_dropdown")
                create_refresh_button(sd_checkpoints_path, scan_sd_ckpt,
                                      lambda: {"choices": sorted(scan_sd_ckpt())}, "refresh_sd_ckpt")

                textual_inversion_path = gr.Dropdown(label="Textual Inversion",
                                                     choices=sorted(scan_textual_inversion_model()),
                                                     elem_id="textual_inversion_model_dropdown")
                create_refresh_button(textual_inversion_path, scan_textual_inversion_model,
                                      lambda: {"choices": sorted(scan_textual_inversion_model())},
                                      "refresh_textual_inversion_model")
            with FormRow(elem_id="model_upload_form_row_02"):
                lora_path = gr.Dropdown(label="LoRA model", choices=sorted(scan_lora_model()),
                                        elem_id="lora_model_dropdown")
                create_refresh_button(lora_path, scan_lora_model,
                                      lambda: {"choices": sorted(scan_lora_model())}, "refresh_lora_model", )

                controlnet_model_path = gr.Dropdown(label="ControlNet model",
                                                    choices=sorted(scan_control_model()),
                                                    elem_id="controlnet_model_dropdown")
                create_refresh_button(controlnet_model_path, scan_control_model,
                                      lambda: {"choices": sorted(scan_control_model())},
                                      "refresh_controlnet_models")
            with FormRow(elem_id="model_upload_form_row_03"):
                hypernetwork_path = gr.Dropdown(label="Hypernetwork", choices=sorted(scan_hypernetwork_model()),
                                                elem_id="hyper_model_dropdown")
                create_refresh_button(hypernetwork_path, scan_hypernetwork_model,
                                      lambda: {"choices": sorted(scan_hypernetwork_model())},
                                      "refresh_hyper_models")

                vae_path = gr.Dropdown(label="VAE", choices=sorted(scan_vae_model()),
                                       elem_id="vae_model_dropdown")
                create_refresh_button(vae_path, scan_vae_model, lambda: {"choices": sorted(scan_vae_model())},
                                      "refresh_vae_models")

            with gr.Row():
                model_update_button = gr.Button(value="Upload Models to Cloud", variant="primary",
                                                elem_id="sagemaker_model_update_button", size=(200, 50))
                model_update_button.click(_js="model_update",
                                          fn=sagemaker_ui.sagemaker_upload_model_s3,
                                          inputs=[sd_checkpoints_path, textual_inversion_path, lora_path,
                                                  hypernetwork_path, controlnet_model_path, vae_path],
                                          outputs=[test_connection_result, sd_checkpoints_path,
                                                   textual_inversion_path, lora_path, hypernetwork_path,
                                                   controlnet_model_path, vae_path])

        with gr.Column(variant="panel"):
            gr.HTML(value="<b>Upload Model to S3 from My Computer</b>")
            gr.HTML(value="Refresh to select the model to upload to S3")
            with FormRow(elem_id="model_upload_local_form_row_01"):
                model_type_drop_down = gr.Dropdown(label="Model Type",
                                                   choices=["SD Checkpoints", "Textual Inversion", "LoRA model",
                                                            "ControlNet model", "Hypernetwork", "VAE"],
                                                   elem_id="model_type_ele_id")
                model_type_hiden_text = gr.Textbox(elem_id="model_type_value_ele_id", visible=False)

                def change_model_type_value(model_type: str):
                    model_type_hiden_text.value = model_type
                    return model_type

                model_type_drop_down.change(fn=change_model_type_value, _js="getModelTypeValue",
                                            inputs=[model_type_drop_down], outputs=model_type_hiden_text)
                file_upload_html_component = gr.HTML(
                    """
                    <div class="lg svelte-1ipelgc">
                        <div class="lg svelte-1ipelgc">
                            <input type="file" 
                                   class="lg secondary gradio-button svelte-1ipelgc" 
                                   id="file-uploader" 
                                   multiple onchange="showFileName(event)" style="width:100%" />
                            </div>
                        </div>
                    """
                )
            with FormRow(elem_id="model_upload_local_form_row_02"):
                hidden_bind_html = gr.HTML(elem_id="hidden_bind_upload_files",
                                           value="<div id='hidden_bind_upload_files_html'></div>")
            with FormRow(elem_id="model_upload_local_form_row_03"):
                upload_label = gr.HTML(label="upload process", elem_id="progress-bar")
                upload_percent_label = gr.HTML(label="upload percent process", elem_id="progress-percent")
            with gr.Row():
                model_update_button_local = gr.Button(value="Upload Models to Cloud", variant="primary",
                                                      elem_id="sagemaker_model_update_button_local",
                                                      size=(200, 50))
                model_update_button_local.click(_js="uploadFiles",
                                                fn=sagemaker_ui.sagemaker_upload_model_s3_local,
                                                # inputs=[sagemaker_ui.checkpoint_info],
                                                outputs=[upload_label]
                                                )

    return upload_tab


def sagemaker_endpoint_tab():
    with gr.Column() as sagemaker_tab:
        gr.HTML(value="<b>Deploy New SageMaker Endpoint</b>")

        with gr.Column(variant="panel"):
            default_table = """
<table style="width:100%; border: 1px solid black; border-collapse: collapse;">
  <tr>
    <th style="border: 1px solid grey; padding: 15px; text-align: left; background-color: #f2f2f2;" colspan="2">Default SageMaker Endpoint Config</th>
  </tr>
  <tr>
    <td style="border: 1px solid grey; padding: 15px; text-align: left;"><b>Instance Type: </b></td>
    <td style="border: 1px solid grey; padding: 15px; text-align: left;">ml.g5.2xlarge</td>
  </tr>
  <tr>
    <td style="border: 1px solid grey; padding: 15px; text-align: left;"><b>Instance Count</b></td>
    <td style="border: 1px solid grey; padding: 15px; text-align: left;">1</td>
  </tr>
  <tr>
    <td style="border: 1px solid grey; padding: 15px; text-align: left;"><b>Automatic Scaling</b></td>
    <td style="border: 1px solid grey; padding: 15px; text-align: left;">yes(range:0-1)</td>
  </tr>

</table>
                    """
            gr.HTML(value=default_table)
            # instance_type_dropdown =
            #   gr.Dropdown(label="SageMaker Instance Type", choices=async_inference_choices, elem_id="sagemaker_inference_instance_type_textbox", value="ml.g4dn.xlarge")
            # instance_count_dropdown =
            #   gr.Dropdown(label="Please select Instance count", choices=["1","2","3","4"], elem_id="sagemaker_inference_instance_count_textbox", value="1")
            endpoint_advance_config_enabled = gr.Checkbox(
                label="Advanced Endpoint Configuration", value=False, visible=True
            )
            with gr.Row(visible=False) as filter_row:
                endpoint_name_textbox = gr.Textbox(value="", lines=1, placeholder="custome endpoint name ",
                                                   label="Specify Endpoint Name", visible=True)
                instance_type_dropdown = gr.Dropdown(label="Instance Type", choices=async_inference_choices,
                                                     elem_id="sagemaker_inference_instance_type_textbox",
                                                     value="ml.g5.2xlarge")
                instance_count_dropdown = gr.Dropdown(label="Max Instance count",
                                                      choices=["1", "2", "3", "4", "5", "6"],
                                                      elem_id="sagemaker_inference_instance_count_textbox",
                                                      value="1")
                autoscaling_enabled = gr.Checkbox(
                    label="Enable Autoscaling (0 to Max Instance count)", value=True, visible=True
                )

            sagemaker_deploy_button = gr.Button(value="Deploy", variant='primary',
                                                elem_id="sagemaker_deploy_endpoint_buttion")
            sagemaker_deploy_button.click(sagemaker_ui.sagemaker_deploy,
                                          _js="deploy_endpoint",
                                          inputs=[endpoint_name_textbox, instance_type_dropdown,
                                                  instance_count_dropdown, autoscaling_enabled],
                                          outputs=[test_connection_result])

        def toggle_new_rows(checkbox_state):
            if checkbox_state:
                return gr.update(visible=True)
            else:
                return gr.update(visible=False)

        endpoint_advance_config_enabled.change(
            fn=toggle_new_rows,
            inputs=endpoint_advance_config_enabled,
            outputs=filter_row
        )

        with gr.Column(title="Delete SageMaker Endpoint", variant='panel'):
            gr.HTML(value="<u><b>Delete SageMaker Endpoint</b></u>")
            with gr.Row():
                sagemaker_endpoint_delete_dropdown = gr.Dropdown(choices=sagemaker_ui.sagemaker_endpoints,
                                                                 multiselect=True,
                                                                 label="Select Cloud SageMaker Endpoint")
                modules.ui.create_refresh_button(sagemaker_endpoint_delete_dropdown,
                                                 sagemaker_ui.update_sagemaker_endpoints,
                                                 lambda: {"choices": sagemaker_ui.sagemaker_endpoints},
                                                 "refresh_sagemaker_endpoints_delete")

            sagemaker_endpoint_delete_button = gr.Button(value="Delete", variant='primary',
                                                         elem_id="sagemaker_endpoint_delete_button")
            sagemaker_endpoint_delete_button.click(sagemaker_ui.sagemaker_endpoint_delete,
                                                   _js="delete_sagemaker_endpoint",
                                                   inputs=[sagemaker_endpoint_delete_dropdown],
                                                   outputs=[test_connection_result])

        return sagemaker_tab


def dataset_tab():
    with gr.Row() as dt:
        with gr.Column(variant='panel'):
            gr.HTML(value="<u><b>Create a Dataset</b></u>")

            def upload_file(files):
                file_paths = [file.name for file in files]
                return file_paths

            file_output = gr.File()
            upload_button = gr.UploadButton("Click to Upload a File", file_types=["image", "video"],
                                            file_count="multiple")
            upload_button.upload(fn=upload_file, inputs=[upload_button], outputs=[file_output])

            def create_dataset(files, dataset_name, dataset_desc):
                logger.debug(dataset_name)
                dataset_content = []
                file_path_lookup = {}
                for file in files:
                    orig_name = file.name.split(os.sep)[-1]
                    file_path_lookup[orig_name] = file.name
                    dataset_content.append(
                        {
                            "filename": orig_name,
                            "name": orig_name,
                            "type": "image",
                            "params": {}
                        }
                    )

                payload = {
                    "dataset_name": dataset_name,
                    "content": dataset_content,
                    "params": {
                        "description": dataset_desc
                    }
                }

                url = get_variable_from_json('api_gateway_url') + '/dataset'
                api_key = get_variable_from_json('api_token')

                raw_response = requests.post(url=url, json=payload, headers={'x-api-key': api_key})
                raw_response.raise_for_status()
                response = raw_response.json()

                logger.info(f"Start upload sample files response:\n{response}")
                for filename, presign_url in response['s3PresignUrl'].items():
                    file_path = file_path_lookup[filename]
                    with open(file_path, 'rb') as f:
                        response = requests.put(presign_url, f)
                        logger.info(response)
                        response.raise_for_status()

                payload = {
                    "dataset_name": dataset_name,
                    "status": "Enabled"
                }

                raw_response = requests.put(url=url, json=payload, headers={'x-api-key': api_key})
                raw_response.raise_for_status()
                logger.debug(raw_response.json())
                return f'Complete Dataset {dataset_name} creation', None, None, None, None

            dataset_name_upload = gr.Textbox(value="", lines=1, placeholder="Please input dataset name",
                                             label="Dataset Name", elem_id="sd_dataset_name_textbox")
            dataset_description_upload = gr.Textbox(value="", lines=1,
                                                    placeholder="Please input dataset description",
                                                    label="Dataset Description",
                                                    elem_id="sd_dataset_description_textbox")
            create_dataset_button = gr.Button("Create Dataset", variant="primary",
                                              elem_id="sagemaker_dataset_create_button")  # size=(200, 50)
            dataset_create_result = gr.Textbox(value="", label="Create Result", interactive=False)
            create_dataset_button.click(
                fn=create_dataset,
                inputs=[upload_button, dataset_name_upload, dataset_description_upload],
                outputs=[
                    dataset_create_result,
                    dataset_name_upload,
                    dataset_description_upload,
                    file_output,
                    upload_button
                ],
                show_progress=True
            )

        with gr.Column(variant='panel'):
            gr.HTML(value="<u><b>Browse a Dataset</b></u>")

            with gr.Row():
                global cloud_datasets
                cloud_datasets = get_sorted_cloud_dataset()

                cloud_dataset_name = gr.Dropdown(
                    label="Dataset From Cloud",
                    choices=[d['datasetName'] for d in cloud_datasets],
                    elem_id="cloud_dataset_dropdown",
                    type="index",
                    info="select datasets from cloud"
                )

                def refresh_datasets():
                    global cloud_datasets
                    cloud_datasets = get_sorted_cloud_dataset()
                    return cloud_datasets

                def refresh_datasets_dropdown():
                    global cloud_datasets
                    cloud_datasets = get_sorted_cloud_dataset()
                    return {"choices": [d['datasetName'] for d in cloud_datasets]}

                create_refresh_button(
                    cloud_dataset_name,
                    refresh_datasets,
                    refresh_datasets_dropdown,
                    "refresh_cloud_dataset",
                )
            with gr.Row():
                dataset_s3_output = gr.Textbox(label='dataset s3 location', show_label=True,
                                               type='text').style(show_copy_button=True)
            with gr.Row():
                dataset_des_output = gr.Textbox(label='dataset description', show_label=True, type='text')
            with gr.Row():
                dataset_gallery = gr.Gallery(
                    label="Dataset images", show_label=False, elem_id="gallery",
                ).style(columns=[2], rows=[2], object_fit="contain", height="auto")

                def get_results_from_datasets(dataset_idx):
                    ds = cloud_datasets[dataset_idx]

                    url = f"{get_variable_from_json('api_gateway_url')}/dataset/{ds['datasetName']}/data"
                    api_key = get_variable_from_json('api_token')
                    raw_response = requests.get(url=url, headers={'x-api-key': api_key})
                    raw_response.raise_for_status()
                    # todo: the s3 presign url is not ready as content type to img
                    dataset_items = [(item['preview_url'], item['key']) for item in
                                     raw_response.json()['data']]
                    return ds['s3'], ds['description'], dataset_items

                cloud_dataset_name.select(fn=get_results_from_datasets, inputs=[cloud_dataset_name],
                                          outputs=[dataset_s3_output, dataset_des_output, dataset_gallery])

    return dt


def update_connect_config(api_url, api_token):
    # Check if api_url ends with '/', if not append it
    if not api_url.endswith('/'):
        api_url += '/'

    save_variable_to_json('api_gateway_url', api_url)
    save_variable_to_json('api_token', api_token)
    global api_gateway_url
    api_gateway_url = get_variable_from_json('api_gateway_url')
    global api_key
    api_key = get_variable_from_json('api_token')
    sagemaker_ui.init_refresh_resource_list_from_cloud()
    return "Setting updated"


def test_aws_connect_config(api_url, api_token):
    update_connect_config(api_url, api_token)
    api_url = get_variable_from_json('api_gateway_url')
    api_token = get_variable_from_json('api_token')
    if not api_url.endswith('/'):
        api_url += '/'
    target_url = f'{api_url}inference/test-connection'
    headers = {
        "x-api-key": api_token,
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(target_url,
                                headers=headers)  # Assuming sagemaker_ui.server_request is a wrapper around requests
        response.raise_for_status()  # Raise an exception if the HTTP request resulted in an error
        r = response.json()
        return "Successfully Connected"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error: Failed to get server request. Details: {e}")
        return "failed to connect to backend server, please check the url and token"
