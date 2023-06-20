# img2img Guide
You can open the **img2img tab** and use the original region along with the **Amazon SageMaker Inference** to perform inference on the cloud.


## img2img use guide
### Standard process for different functional labels in img2img

1. Navigate to tab **img2img**, open pannel **Amazon SageMaker Inference**.
2. Input parameters for inference. The same as local inferance, you could edit parameters in native fields for prompts, negative prompts, sampling parameters, inference parameters and etc. For functions **img2img**, **sketch**, **inpaint**, **inpaint sketch** and **inpaint upload**, you could upload and modify images in the native way.
3. Select the inference endpoint. Click the refresh button on the right side of **Select Cloud SageMaker Endpoint** to choose an inference endpoint that is in the **InService** state.

    !!! Important "Notice" 
        This field is mandatory. If you choose an endpoint that is in any other state or leave it empty, an error will occur when you click **Generate on Cloud** to initiate cloud-based inference.

4. Fresh and select **Stable Diffusion Checkpoint** (required single select) and other extra models needed in **Extra Networks for Cloud Inference** (optional, multi-selection allowed).
5. Click**Generate on Cloud**。
6. Check inference result. Fresh and select the top option among **Inference Job ID** dropdown list. The **Output** section in the top-right area of the img2img tab will display the results of the inference once completed, including the generated images, prompts, and inference parameters. Based on this, you can perform subsequent workflows such as clicking **Save** or **Send to extras**.
> **Note:** The list is sorted in reverse chronological order based on the inference time, with the most recent inference task appearing at the top. Each record is named in the format of **inference time -> job type(txt2img/img2img/interrogate_clip/interrogate_deepbooru) -> inference status(succeed/in progress/fail) ->inference id**.


### img2img label

1. Upload the original image to **img2img** and enter prompts, for example *black flower*.
![img2img raw](../images/img2img_tab.png)

2. Click **Generate on Cloud**, and select corresponding **Inference Job ID**, the generated image will present on the right **Output** session. 
![img2img result](../images/img2img_result.png)



### Sketch label

1. Start **webui** with ‘--gradio-img2img-tool color-sketch’ on the command line，upload the whiteboard background image to the **Sketch tab**.
![sketch](../images/sketch_raw.png)

2. Use a brush to draw the corresponding sketch and prepare prompt words, for example **flower**.
![sketch paint](../images/sketch_paint.png)

3. Click **Generate on Cloud**, and select corresponding **Inference Job ID**, the generated image will present on the right **Output** session.
![sketch result](../images/sketch_result.png)


### Inpaint label

1. Upload original image to **Inpaint** label and prepare prompts, for example *watermelon*。
![inpaint](../images/inpaint_tab.png)

2. Establish masks with brushes and prepare prompts, for example *watermelon*。
![inpaint_inpaint](../images/inpaint_inpaint.png)

3. Click **Generate on Cloud**, and select corresponding **Inference Job ID**, the generated image will present on the right **Output** session.
![inpaint result](../images/inpaint_result.png)


### Inpatnt Sketch label

1. Start **webui** with ‘--gradio-img2img-tool color-sketch’，and upload original image into **Inpaint Sketch**label with prompts, for example *candy*。
![inpaint_sketch](../images/inpaint_sketch_tab.png)

2. Establish masks with brushes.
![inpaint_sketch_inpaint](../images/inpaint_sketch_inpaint.png)

3. Click **Generate on Cloud**, and select corresponding **Inference Job ID**, the generated image will present on the right **Output** session.
![inpaint_sketch result](../images/inpaint_sketch_result.png)


### Inpatnt Upload label

1. Upload original image and mask image to **Inpaint Upload** label and prepare prompts, for example *large eyes*.
![inpaint_upload](../images/inpaint_upload_tab.png)

2. Click **Generate on Cloud**, and select corresponding **Inference Job ID**, the generated image will present on the right **Output** session.
![inpaint_upload result](../images/inpaint_upload_result.png)



### Interrogate clip/deepbooru

1. Navigate to **img2img** tab，open **Amazon SageMaker Inference** pannel.
2. Interrogate only need to upload image to **img2img** tab.
![img2img tab](../images/clip_tab.png)

3. Select inference endpoint. Click refresh button on the right of **Select Cloud SageMaker Endpoint**, select one inference endpoint in **InService** state.
4. Click **Interrogate CLIP on cloud** or **Interrogate DeepBooru on cloud**.
5. Check inference result. Refresh the dropdown list of **Inference Job JDs**, check the topmost **Inference Job ID** that match the inference submission timestamp to review the inference result.
![interrogate generate results](../images/clip.png)

### Continous Inference Scenario

1. Following the **General Inference Scenario**, complete the parameter inputs and click **Generate on Cloud** to submit the initial inference task.
2. Wait for the appearance of a new **Inference ID** in the right-side **Output** section.
3. Once the new Inference ID appears, you can proceed to click **Generate on Cloud** again for the next inference task.
![continuous generate results](../images/continue-inference.png)