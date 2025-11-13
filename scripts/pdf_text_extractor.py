import pymupdf
import os
import sys



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from  utils.config_loader import ConfigLoader  
# Charger la config avec ton ConfigLoader
config_loader = ConfigLoader("config.yaml")

# Chemins depuis la config
raw_base_dir = config_loader.get_path(config_loader.config["data"]["raw_dir"])
processed_base_dir = config_loader.get_path(config_loader.config["data"]["processed_dir"])
bourgogne_raw_dir = config_loader.config['scraping']['regions']['bourgogne_franche_comte']['output_dir_pase_path']

# Chemin complet 
raw_full_path = config_loader.get_path(bourgogne_raw_dir)

for root, dirs, files in os.walk(raw_full_path):
    for file in files:
        if file.endswith('.pdf'):
            pdf_path = os.path.join(root, file)

            relative_path = os.path.relpath(root, raw_full_path)
            # Utiliser seulement 'bourgogne_franche_comte' pour éviter la duplication
            output_dir = os.path.join(processed_base_dir, 'bourgogne_franche_comte', relative_path)
            os.makedirs(output_dir, exist_ok=True)
            txt_filename = os.path.splitext(file)[0] + '.txt'
            output_path = os.path.join(output_dir, txt_filename) 
            
            with pymupdf.open(pdf_path) as doc:
                with open(output_path, "w", encoding="utf8") as out:
                    for page in doc:
                        text = page.get_text()
                        out.write(text)
                        out.write("\n")
                        out.write("\n")

            print(f"Texte extrait sauvegardé dans {output_path}")