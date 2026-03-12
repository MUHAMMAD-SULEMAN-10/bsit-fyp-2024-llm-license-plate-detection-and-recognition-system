import os
import sys
import zipfile
import io
import requests
import json
if not os.path.exists("output"):
    os.mkdir("output")
class VisionLanguageModel:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.prompts = [
            "<CAPTION>",
            "<DETAILED_CAPTION>",
            "<MORE_DETAILED_CAPTION>",
            "<OD>",
            "<DENSE_REGION_CAPTION>",
            "<REGION_PROPOSAL>",
            "<CAPTION_TO_PHRASE_GROUNDING>A black and brown dog is laying on a grass field.",
            "<REFERRING_EXPRESSION_SEGMENTATION>a black and brown dog",
            "<REGION_TO_SEGMENTATION><loc_312><loc_168><loc_998><loc_846>",
            "<OPEN_VOCABULARY_DETECTION>a black and brown dog",
            "<REGION_TO_CATEGORY><loc_312><loc_168><loc_998><loc_846>",
            "<REGION_TO_DESCRIPTION><loc_312><loc_168><loc_998><loc_846>",
            "<OCR>",
            "<OCR_WITH_REGION>"
        ]

    def upload_asset(self, file_path, description):
        """
        Uploads an asset to the NVCF API and returns the asset ID.
        :param file_path: Path to the image file.
        :param description: Description of the asset.
        :return: asset_id (str)
        """
        with open(file_path, "rb") as image_file:
            authorize = requests.post(
                "https://api.nvcf.nvidia.com/v2/nvcf/assets",
                headers={
                    "Authorization": self.headers["Authorization"],
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                json={"contentType": "image/jpeg", "description": description},
                timeout=30
            )
            authorize.raise_for_status()

            response = requests.put(
                authorize.json()["uploadUrl"],
                data=image_file,
                headers={
                    "x-amz-meta-nvcf-asset-description": description,
                    "content-type": "image/jpeg",
                },
                timeout=300
            )
            response.raise_for_status()
            return authorize.json()["assetId"]

    def generate_content(self, task_id, asset_id):
        """
        Generates the content string for the API request.
        :param task_id: ID of the task to perform.
        :param asset_id: Asset ID of the uploaded image.
        :return: content string (str)
        """
        if task_id < 0 or task_id >= len(self.prompts):
            raise ValueError(f"task_id should be within [0, {len(self.prompts)-1}]")

        prompt = self.prompts[task_id]
        return f'{prompt}<img src="data:image/jpeg;asset_id,{asset_id}" />'
    

    def extract_labels_from_response(self,extracted_data):
        """
        Extracts labels from the extracted JSON data within the response and removes unwanted prefixes like <S>.
        """
        labels = []

        # Iterate through the extracted data
        for file_name, content in extracted_data.items():
            if file_name.endswith('.response'):
                # Parse the JSON content
                data = json.loads(content)
                
                # Check if the required data exists and extract the labels
                if 'choices' in data and len(data['choices']) > 0:
                    message = data['choices'][0].get('message', {})
                    entities = message.get('entities', {})
                    raw_labels = entities.get('labels', [])
                    
                    # Remove the '<S>' string from each label
                    clean_labels = [label.replace('</s>', '') for label in raw_labels]
                    labels.extend(clean_labels)
                    
        return labels
    def extract_from_response_content(self, content):
        """
        Extracts files directly from the response content (zip format) in memory.
        Returns data from the extracted files.
        """
        extracted_data = {}

        try:
            print("content", content)

            # Use BytesIO to treat the content as a file-like object
            with zipfile.ZipFile(io.BytesIO(content), 'r') as zip_ref:
                # Iterate over the files in the zip
                for file_name in zip_ref.namelist():
                    try:
                        with zip_ref.open(file_name) as file:
                            file_content = file.read()
                            extracted_data[file_name] = file_content.decode('utf-8', errors='ignore')  # Decoding content to text
                    except Exception as e:
                        print(f"Error reading file {file_name}: {e}")

        except zipfile.BadZipFile:
            print("Error: The provided content is not a valid ZIP file.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        return extracted_data
        

    def process_task(self, image_path, output_dir, task_id):
        """
        Processes the task: uploads the image, generates content, and sends the API request.
        :param image_path: Path to the image file.
        :param output_dir: Directory to save the results.
        :param task_id: ID of the task to perform.
        """
        # Upload the asset
        asset_id = self.upload_asset(image_path, "Test Image")

        # Generate the content
        content = self.generate_content(task_id, asset_id)

        # Prepare the API request payload
        inputs = {
            "messages": [{
                "role": "user",
                "content": content
            }]
        }

        # Add asset references to headers
        headers = self.headers.copy()
        headers.update({
            "NVCF-INPUT-ASSET-REFERENCES": asset_id,
            "NVCF-FUNCTION-ASSET-IDS": asset_id
        })

        # Send the request
        response = requests.post(self.base_url, headers=headers, json=inputs)
        # response.raise_for_status()
        # Extract directly from response.content
        extracted_data = self.extract_from_response_content(response.content)
        final_result = self.extract_labels_from_response(extracted_data)
        print(final_result)


        # Return the extracted data
        return {
            "response": True,
            "extracted_data": final_result  # Include extracted data here
        }

