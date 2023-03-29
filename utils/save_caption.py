import torch
import os
import argparse
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer


max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(i_image):
  images = []

  images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--img_dir', default='./data/MNRE_images/')
  parser.add_argument('--caption_file', default='./ner_img/caption_mre.pt', type=str)
  parser.add_argument('--pretrained_model', default='../pretrained_model/vit-gpt2-image-captioning')

  args = parser.parse_args()

  model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_model)
  feature_extractor = ViTFeatureExtractor.from_pretrained(args.pretrained_model)
  tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  model.eval()


  img_id_list = os.listdir(args.img_dir)
  count = 0
  img_caption_dict = {}

  for i, img_id in enumerate(img_id_list):
      image = None
      try:
          img_path = os.path.join(args.img_dir, img_id)
          image = Image.open(img_path).convert('RGB')
          pred = predict_step(image)[0]
          img_caption_dict[img_id] = pred
      except:
          img_caption_dict[img_id] = "There are nothing in the image"

  print("count:", count)
  torch.save(img_caption_dict, args.caption_file)



