import json
import re 

#Clean MIMIC.json inference results
def clean_mimic_json(messy_json, cleaned_output):
    with open(messy_json, 'r') as f:
        messy_data = json.load(f)

    clean_data = []
    for image_id, captions in messy_data.items():
        # Extract the image_id without the file extension
        image_id_clean = image_id.split('.')[0]
        # Join the captions list into a single string
        caption_clean = ' '.join(captions)

        clean_item = {
            "image_id": image_id_clean,
            "caption": caption_clean
        }
        
        clean_data.append(clean_item)

    # Write the cleaned data to clean.json
    with open(cleaned_output, 'w') as outfile:
        json.dump(clean_data, outfile, indent=2)


#Clean VQA.json inference results
def clean_vqa_json(messy_json, cleaned_output):
    # Read the messy JSON data from a file
    with open(messy_json, "r") as file:
        messy_json = json.load(file)

    # Create a new organized JSON structure
    organized_json = {}

    for key, values in messy_json.items():
        organized_json[key] = []
        for value in values:
            organized_json[key].append({
                "question": value["question"],
                "answer": value["answer"]
            })

    with open(cleaned_output, "w") as outfile:
        json.dump(organized_json, outfile, indent=4)



#Clean detetcion inference output
def clean_detection_json(messy_json, cleaned_output):

    # Read data from NLST_inference_result.json
    with open(messy_json, "r") as input_file:
        input_json = json.load(input_file)

    # Initialize an empty list to store the organized data
    organized_data = []

    # Iterate through the existing JSON data
    for key, value in input_json.items():
        if value and isinstance(value, list) and len(value) > 0:
            caption = value[0]
            objects_match = caption.split("<p>")
            if len(objects_match) == 2:
                object_part = objects_match[1].split("</p>")[0].strip()
            else:
                object_part = ""
            
            bbox_match = re.findall(r'<(\d+)>', caption)
            
            if object_part and bbox_match and len(bbox_match) == 4:
                key_part = key.split(".png")[0]
                bbox_values = [float(val) for val in bbox_match]

                organized_item = {
                    "key": key_part,
                    "objects": [object_part],
                    "bbox": [bbox_values],
                }

                organized_data.append(organized_item)

    with open(cleaned_output, "w") as output_file:
        json.dump(organized_data, output_file, indent=4)