import streamlit as st

import torch 
import cv2
import wget
import numpy as np 
import tensorflow as tf 



def main():
    RESNET_LAST_ONLY = False
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    COMPARTMENTS = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

    st.title('RobyNet neural network for trash classification')

    st.write('This app shows the demo of how we are going to streamline the process of cluttering the trash to distinc recycle compartments using computer vision')
    #st.subheader('You may find all the details as below')
    
    st.write('----------------------------------')

    st.subheader('Here you may upload or take a picture to test the neural network')
    st.warning('Testing accuracy is 93.42% with the WasteNet dataset')
    image = st.file_uploader("Upload the image here")
    #model = PreTrainedResNet(len(class_names),RESNET_LAST_ONLY)
    #model.cuda()
    #model.load_state_dict(torch.load('resnet_roby_state.pt'))
    model_pt = download()
    #for param_tensor in model.state_dict():
    #    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    if image is not None:
        model_pt.eval()
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        out = cv2.resize(img, (343, 343))
    
        show_img = cv2.resize(img, (512,512))
        st.image(show_img, caption= 'This is the uploaded image', channels="BGR")

        #img_input = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = out/255
        model_input = out.reshape(1,3,343,343)
        
        out_tensor = torch.from_numpy(model_input)
        outputs = model_pt.forward(out_tensor.float())
        _, preds = torch.max(outputs.data, 1)

        predlabels = preds.cpu().numpy()
        st.warning("Here is what neural network thinks: ")
        st.write('')
        #labels_num = labels.cpu().numpy()
        softmax_outs = outputs.softmax(dim=1)
        softmax_np = softmax_outs.cpu().detach().numpy()
        percentages = softmax_np/np.sum(softmax_np)
        #print(percentages)
        col0, col1, col2, col3, col4, col5 = st.columns(6)

        col0.metric(COMPARTMENTS[0], str(round(percentages[0][0]*100,1))+ " %")
        col1.metric(COMPARTMENTS[1], str(round(percentages[0][1]*100,1))+ " %")
        col2.metric(COMPARTMENTS[2], str(round(percentages[0][2]*100,1))+ " %")
        col3.metric(COMPARTMENTS[3], str(round(percentages[0][3]*100,1))+ " %")
        col4.metric(COMPARTMENTS[4], str(round(percentages[0][4]*100,1))+ " %")
        col5.metric(COMPARTMENTS[5], str(round(percentages[0][5]*100,1))+ " %")
        #print(predlabels)
        st.success('The final class is ' + COMPARTMENTS[predlabels[0]])
    st.write("----------------------")
    st.subheader('Here is some image examples that has been used to train the neural network')
    eg_img1, eg_img2 = st.columns(2)
    eg_img3, eg_img4 = st.columns(2)

    eg_img1.image('img1.jpg', caption='Trash')
    eg_img2.image('img2.jpg', caption='Glass')
    eg_img3.image('img3.jpg', caption='Metal')
    eg_img4.image('img4.jpg', caption='Paper')


        

@st.cache(allow_output_mutation=True, show_spinner=False)
def download():
    img1 = 'img1.jpg'
    img2 = 'img2.jpg'
    img3 = 'img3.jpg'
    img4 = 'img4.jpg'
    filename_model = 'resnet_rb.pt'
    img1_link = 'https://www.dropbox.com/s/bcyvbrvvad24bzi/trash28.jpg?dl=1'
    img2_link = 'https://www.dropbox.com/s/s6usccgjng4yvio/glass17.jpg?dl=1'
    img3_link = 'https://www.dropbox.com/s/168rx6kqhten493/metal89.jpg?dl=1'
    img4_link = 'https://www.dropbox.com/s/4t515zwgomhea0n/paper121.jpg?dl=1'
    model_link = 'https://www.dropbox.com/s/jzx0vno3w9aht9e/resnet_roby.pt?dl=1'
    
    wget.download(img1_link, img1)
    wget.download(img2_link, img2)
    wget.download(img3_link, img3)
    wget.download(img4_link, img4)
    wget.download(model_link, filename_model)
    model_pt = torch.load('resnet_rb.pt')
    return model_pt
    


if __name__ == "__main__":
  main()