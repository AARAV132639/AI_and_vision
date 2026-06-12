import cv2
import mediapipe as mp
import pandas as pd
import os 


mp_hands = mp.solutions.hands.Hands(static_image_mode= True, max_num_hands=1)

def extract_landmarks(image_path):

    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Could not read image{image_path}")
        return None 

    results = mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:

        #Flatten the 21 landmarks (x,y,z) into list of 63 value

        landmarks = []

        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x,lm.y,lm.z])
         
        return landmarks
    
    else:
        print(f"No hand detected ")

    return None


def process_directory(base_path):

    print(f"Scanning directory: {base_path}")
    data = []

    #iterating over files directly

    for img_name in os.listdir(base_path):

        if img_name.endswith(('.jpg','.jpeg','.png')): #Fileter for image
            img_path = os.path.join(base_path,img_name)

            # logic: extract label from fielname

            label = img_name.split('_')[0].split('(')[0]

            landmarks = extract_landmarks(img_path)

            if landmarks:
                landmarks.append(label)
                data.append(landmarks)
                print(f"Coolected:{label}from{img_name}")
            
            else:
                print(f"Failed {img_name}")
    
    return data 


# Training for both
train_data = process_directory('data/raw/images/train')
test_data = process_directory('data/raw/images/val')

# save to csv
pd.DataFrame(train_data).to_csv('train_landmarks.csv', index= False, header = False)
pd.DataFrame(test_data).to_csv('test_landmarks.csv', index= False, header = False)

mp_hands.close()

print("Landmark extraction complete!")


