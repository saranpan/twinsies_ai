import torch
import torchvision.transforms as T
from facenet_pytorch import MTCNN # For face alignment (crop + position)
from facenet_pytorch import InceptionResnetV1 # For embedding face to vector
import streamlit as st
from streamlit_option_menu import option_menu
from utils import compute_l2_dist, siamese_network, pipeline, get_random_file
from PIL import Image, UnidentifiedImageError
import io
import pickle
import os

# Read dictionary pkl file (database)
with open('emb_database/image_tensor.pkl', 'rb') as fp:
    img_database = pickle.load(fp)
    
# Seuup some stuffs including transforms, and models
tensor_to_img = T.ToPILImage()
img_to_tensor = T.ToTensor() # เกบไว้ตอนเชค destination path

mtcnn = MTCNN(image_size=160, margin=0) # face align
resnet = InceptionResnetV1(pretrained='vggface2',classify=False).eval() # face to emb

# Setup page title
st.set_page_config(page_title="Twinsies AI", page_icon=":full_moon_face:", layout="wide")

selected = option_menu(
        menu_title= None , 
        options=['Predict','Q&A'],
        icons=['bi bi-speedometer2','house'], 
        menu_icon="cast",
        default_index = 0,
        orientation="horizontal")

if selected == 'Predict':
    with st.sidebar:
        input_method = st.selectbox('Select Input method', options=['file uploader', 'camera capture'])
        
    st.header('👯 Twinsies AI: Who Do You Look Like in the Celebrity World?')
    st.write("""A web application that uses state-of-the-art technology
            to find your celebrity doppelganger. By uploading a photo of yourself (or any persons) or camera capturing, 
            our app uses the MTCNN algorithm to detect and align your facial features, 
            followed by the FaceNet model to extract your face's unique features. 
            We then compare your features to our extensive celebrity database to find 
            the closest match to your appearance. With Twinsies, you can finally uncover 
            which famous face you most closely resemble.""")

    with st.container():
        upload_col, capture_col = st.columns((1,1))
        if input_method == 'file uploader':
            test_img = st.file_uploader("Upload image", type = ['jpg','jpeg','png'])
        elif input_method == 'camera capture':
            test_img = st.camera_input("Capture image")
    if test_img:
        infer_dct = {}
        test_img = Image.open(io.BytesIO(test_img.getvalue())).convert('RGB')
        
        test_img_tensor, face_prob = pipeline(test_img, mtcnn, resnet)
        if face_prob is None: #None if the image is less than threshold
            st.error("No face were detected")
            st.stop()
            
        with st.spinner('Comparing your face with other celebrity faces ...'): 
            for k, v in img_database.items():
                diff = test_img_tensor - v
                dist = torch.linalg.vector_norm(diff,dim=1)
                dist = torch.mean(dist)
                infer_dct.update({k : dist.item()})

            # Ascending by degree of difference 
            infer_dct = dict(sorted(infer_dct.items(), key=lambda item: item[1]))
            top3_person = list(infer_dct.keys())[:3]
            top3_diff = list(infer_dct.values())[:3]

        st.balloons()
        _, you, _ = st.columns((1,1,1))
        with you:
            image = test_img.resize((600, 600))
            st.image(image)
            st.write('You')

        rank1, rank2, rank3 = st.columns((1,1,1))
        with rank1:
            rank1_dir = f'database/{top3_person[0]}'
            random_img = get_random_file(rank1_dir)
            image = Image.open(random_img)
            image = image.resize((600, 600))
            st.image(image)
            st.write(f'🥇 {top3_person[0].title()}')
            st.write(f'Degree of difference : {top3_diff[0]:.2f}')

        with rank2:
            rank2_dir = f'database/{top3_person[1]}'
            random_img = get_random_file(rank2_dir)
            image = Image.open(random_img)
            image = image.resize((600, 600))
            st.image(image)
            st.write(f'🥈 {top3_person[1].title()}')
            st.write(f'Degree of difference : {top3_diff[1]:.2f}')

        with rank3:
            rank3_dir = f'database/{top3_person[2]}'
            random_img = get_random_file(rank3_dir)
            image = Image.open(random_img)
            image = image.resize((600, 600))
            st.image(image)
            st.write(f'🥉 {top3_person[2].title()}')
            st.write(f'Degree of difference : {top3_diff[2]:.2f}')

elif selected == 'Q&A':
    
    with st.expander("Which celebrities are included in our app's collection?"):
        st.write("Our Database contain 28 celebrities, 3 images per celebrities")
        st.image(Image.open('web_image/file_tree.png'))
        st.write(f"Update 22/4/23 : Add the owner images of this app too, so it's 29 person xD")

    with st.expander('Can I add more celebrities other than these ?'):
        st.write("""
                Yes!, Here's the following steps
                1. Download folder Twinsies from our github repository Twinsie (Here: https://github.com/saranpan/twinsies_ai)
                """)
        
        st.image(Image.open('web_image/step1.png'))

        st.write("2. Now, select at the folder Database (A folder where we store all celeb images)")
        st.image(Image.open('web_image/step2.png'))

        st.write("3. Now, Create a new folder, new celeb (say saran pan)")
        st.image(Image.open('web_image/step3.png'))

        st.write("4. Now add that person image into that folder (At least one image is required)")
        st.image(Image.open('web_image/step4.png'))

        st.write("5. Run commit.py to embedding images into tensor")
        st.image(Image.open('web_image/step5.png'))

        