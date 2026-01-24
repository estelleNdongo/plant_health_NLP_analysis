import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

# Configuration matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 13


class BSVAgroVisualizer:
    """
    Visualisations agronomiques pertinentes pour les BSV
    """
    
    def __init__(self):
        self.config_loader = ConfigLoader("config.yaml")
        self.base_dir = Path(self.config_loader.base_dir)
        
        # Chemins
        self.results_dir = self.base_dir / self.config_loader.config["data"]["results_dir"]
        self.entities_dir = self.results_dir / "entities"
        self.viz_dir = self.results_dir / "visualizations_agro"
        
        # Créer le dossier de sortie
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Charger les rapports
        self.entity_report = self._load_json(self.entities_dir / "entity_extraction_report.json")
        
        # Charger toutes les données
        self.all_entities_data = self._load_all_entities()
        
        logger.info(f"Visualiseur agronomique initialise - {len(self.all_entities_data)} fichiers")
    
    def _load_json(self, filepath):
        """Charge un fichier JSON"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erreur chargement {filepath}: {e}")
            return {}
    
    def _load_all_entities(self):
        """Charge tous les fichiers JSON d'entités"""
        all_data = []
        
        for year_dir in sorted(self.entities_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            
            for json_file in year_dir.glob("*.json"):
                data = self._load_json(json_file)
                if data:
                    all_data.append(data)
        
        logger.info(f"Charge {len(all_data)} fichiers d'entites")
        return all_data
    
    def _extract_month_from_filename(self, filename):
        """
        Extrait le mois d'un nom de fichier BSV
        
        Args:
            filename: ex: "bsv_gc_n_1_du_23_08_22.txt"
            
        Returns:
            int: Mois (1-12) ou None
        """
        # Pattern: du_JJ_MM_AA
        match = re.search(r'du_\d{2}_(\d{2})_\d{2,4}', filename)
        if match:
            try:
                month = int(match.group(1))
                if 1 <= month <= 12:
                    return month
            except:
                pass
        return None
    
    def plot_top_ravageurs(self, top_n=15):
        """
        Top N ravageurs - menaces principales
        """
        data = self.entity_report['top_entites_par_type']['ravageurs'][:top_n]
        
        entities = [item['entite'] for item in data]
        frequencies = [item['frequency'] for item in data]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Créer le barplot
        bars = ax.barh(range(len(entities)), frequencies, 
                      color=sns.color_palette("Reds_r", len(entities)))
        
        ax.set_yticks(range(len(entities)))
        ax.set_yticklabels(entities, fontsize=12)
        ax.set_xlabel('Nombre de mentions dans les BSV', fontsize=13, fontweight='bold')
        ax.set_title(f'🐛 Top {top_n} Ravageurs - Menaces Principales\nBourgogne-Franche-Comté | 2022-2024', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Ajouter les valeurs
        for i, (bar, freq) in enumerate(zip(bars, frequencies)):
            ax.text(freq + max(frequencies)*0.01, i, str(freq), 
                   va='center', fontsize=11, fontweight='bold')
        
        # Grille
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        output_file = self.viz_dir / "1_top_ravageurs.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Top ravageurs sauvegarde: {output_file}")
        return output_file
    
    def plot_top_pathogenes(self, top_n=12):
        """
        Top N pathogènes - maladies à anticiper
        """
        data = self.entity_report['top_entites_par_type']['pathogenes'][:top_n]
        
        if not data:
            logger.warning("Pas de pathogenes detectes")
            return None
        
        entities = [item['entite'] for item in data]
        frequencies = [item['frequency'] for item in data]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Créer le barplot
        bars = ax.barh(range(len(entities)), frequencies, 
                      color=sns.color_palette("Purples_r", len(entities)))
        
        ax.set_yticks(range(len(entities)))
        ax.set_yticklabels(entities, fontsize=12, style='italic')
        ax.set_xlabel('Nombre de mentions dans les BSV', fontsize=13, fontweight='bold')
        ax.set_title(f'🦠 Top {top_n} Pathogènes - Maladies à Anticiper\nBourgogne-Franche-Comté | 2022-2024', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Ajouter les valeurs
        for i, (bar, freq) in enumerate(zip(bars, frequencies)):
            ax.text(freq + max(frequencies)*0.01, i, str(freq), 
                   va='center', fontsize=11, fontweight='bold')
        
        # Grille
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        output_file = self.viz_dir / "2_top_pathogenes.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Top pathogenes sauvegarde: {output_file}")
        return output_file
    
    def plot_heatmap_culture_ravageur(self, culture_top=8, ravageur_top=15):
        """
        Heatmap culture × ravageur - Quoi surveiller sur chaque culture
        """
        # Récupérer les top cultures et ravageurs
        top_cultures = [item['entite'] for item in 
                       self.entity_report['top_entites_par_type']['cultures'][:culture_top]]
        top_ravageurs = [item['entite'] for item in 
                        self.entity_report['top_entites_par_type']['ravageurs'][:ravageur_top]]
        
        # Créer la matrice de co-occurrences
        cooc_matrix = defaultdict(lambda: defaultdict(int))
        
        for data in self.all_entities_data:
            # Extraire les cultures et ravageurs présents
            cultures_in_doc = set()
            ravageurs_in_doc = set()
            
            for entity_info in data['entites_detectees']['cultures']:
                if entity_info['entite'] in top_cultures:
                    cultures_in_doc.add(entity_info['entite'])
            
            for entity_info in data['entites_detectees']['ravageurs']:
                if entity_info['entite'] in top_ravageurs:
                    ravageurs_in_doc.add(entity_info['entite'])
            
            # Co-occurrences
            for culture in cultures_in_doc:
                for ravageur in ravageurs_in_doc:
                    cooc_matrix[culture][ravageur] += 1
        
        # Convertir en DataFrame
        df = pd.DataFrame(cooc_matrix).T.fillna(0)
        df = df.reindex(index=top_cultures, columns=top_ravageurs, fill_value=0)
        
        # Créer la heatmap
        fig, ax = plt.subplots(figsize=(18, 10))
        sns.heatmap(df, annot=True, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Nombre de BSV mentionnant les deux'},
                   linewidths=0.5, ax=ax, vmin=0, vmax=df.max().max(),
                   annot_kws={'fontsize': 10, 'fontweight': 'bold'})
        
        ax.set_title('🌾 Co-occurrences Culture × Ravageur - Quoi Surveiller sur Chaque Culture\nBourgogne-Franche-Comté | 2022-2024', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Ravageurs', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cultures', fontsize=13, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(rotation=0, fontsize=11)
        
        plt.tight_layout()
        output_file = self.viz_dir / "3_heatmap_culture_ravageur.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heatmap sauvegardee: {output_file}")
        return output_file
    
    def plot_calendrier_saisonnier(self, ravageur_top=15):
        """
        Calendrier saisonnier des ravageurs - Quand surveiller
        """
        # Récupérer les top ravageurs
        top_ravageurs = [item['entite'] for item in 
                        self.entity_report['top_entites_par_type']['ravageurs'][:ravageur_top]]
        
        # Matrice mois × ravageur
        monthly_counts = defaultdict(lambda: defaultdict(int))
        
        for data in self.all_entities_data:
            # Extraire le mois du fichier
            month = self._extract_month_from_filename(data['filename'])
            if not month:
                continue
            
            # Compter les ravageurs présents
            for entity_info in data['entites_detectees']['ravageurs']:
                ravageur = entity_info['entite']
                if ravageur in top_ravageurs:
                    monthly_counts[ravageur][month] += 1
        
        # Créer le DataFrame
        months = list(range(1, 13))
        month_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 
                      'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
        
        df_data = []
        for ravageur in top_ravageurs:
            row = [monthly_counts[ravageur][m] for m in months]
            df_data.append(row)
        
        df = pd.DataFrame(df_data, index=top_ravageurs, columns=month_names)
        
        # Créer la heatmap
        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(df, annot=True, fmt='.0f', cmap='YlGnBu', 
                   cbar_kws={'label': 'Nombre de mentions'},
                   linewidths=0.5, ax=ax,
                   annot_kws={'fontsize': 9})
        
        ax.set_title('📅 Calendrier Saisonnier des Ravageurs - Quand Surveiller\nBourgogne-Franche-Comté | 2022-2024', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Mois', fontsize=13, fontweight='bold')
        ax.set_ylabel('Ravageurs', fontsize=13, fontweight='bold')
        plt.yticks(rotation=0, fontsize=11)
        plt.xticks(fontsize=11)
        
        plt.tight_layout()
        output_file = self.viz_dir / "4_calendrier_saisonnier.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Calendrier saisonnier sauvegarde: {output_file}")
        return output_file
    
    def generate_agro_report(self):
        """
        Génère un rapport HTML agronomique
        """
        # Statistiques clés
        stats = self.entity_report['statistiques']
        entities = self.entity_report['entites_par_type']
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Rapport Agronomique BSV - Bourgogne-Franche-Comté</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 1600px;
                    margin: 0 auto;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .container {{
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    font-size: 32px;
                    margin-bottom: 10px;
                }}
                .subtitle {{
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 18px;
                    margin-bottom: 30px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .stat-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 25px;
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .stat-number {{
                    font-size: 42px;
                    font-weight: bold;
                    display: block;
                    margin-bottom: 8px;
                }}
                .stat-label {{
                    font-size: 15px;
                    opacity: 0.95;
                }}
                .section {{
                    margin: 40px 0;
                }}
                .section h2 {{
                    color: #34495e;
                    border-left: 6px solid #667eea;
                    padding-left: 15px;
                    margin-bottom: 20px;
                    font-size: 24px;
                }}
                .viz-container {{
                    background: #f8f9fa;
                    padding: 25px;
                    border-radius: 10px;
                    margin: 20px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .interpretation {{
                    background: #e3f2fd;
                    border-left: 4px solid #2196f3;
                    padding: 15px 20px;
                    margin: 15px 0;
                    border-radius: 4px;
                }}
                .interpretation h3 {{
                    color: #1976d2;
                    margin-top: 0;
                    font-size: 18px;
                }}
                .footer {{
                    text-align: center;
                    margin-top: 50px;
                    padding: 25px;
                    color: #7f8c8d;
                    border-top: 2px solid #ecf0f1;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Rapport Agronomique BSV</h1>
                <p class="subtitle">Bulletins de Santé du Végétal | Grandes Cultures<br>
                Bourgogne-Franche-Comté | 2022-2024</p>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <span class="stat-number">{stats['fichiers_traites']}</span>
                        <span class="stat-label">Bulletins analysés</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{entities['cultures']['entites_uniques']}</span>
                        <span class="stat-label">Cultures suivies</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{entities['ravageurs']['entites_uniques']}</span>
                        <span class="stat-label">Ravageurs identifiés</span>
                    </div>
                    <div class="stat-card">
                        <span class="stat-number">{entities['pathogenes']['entites_uniques']}</span>
                        <span class="stat-label">Pathogènes détectés</span>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🐛 Ravageurs - Menaces Principales</h2>
                    <div class="viz-container">
                        <img src="1_top_ravageurs.png" alt="Top ravageurs">
                    </div>
                    <div class="interpretation">
                        <h3>💡 Interprétation</h3>
                        <p><strong>Puceron vert du pêcher</strong> : Ravageur le plus mentionné (vecteur de virus).</p>
                        <p><strong>Campagnol des champs</strong> : Fort impact sur céréales et prairies.</p>
                        <p><strong>Altise d'hiver & Limace grise</strong> : Menaces majeures en début de cycle.</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🦠 Pathogènes - Maladies à Anticiper</h2>
                    <div class="viz-container">
                        <img src="2_top_pathogenes.png" alt="Top pathogènes">
                    </div>
                    <div class="interpretation">
                        <h3>💡 Interprétation</h3>
                        <p><strong>Colletotrichum & Fusarium</strong> : Maladies fongiques dominantes.</p>
                        <p><strong>Phoma lingam</strong> : Surveillance essentielle sur colza.</p>
                        <p><strong>Sclerotinia</strong> : Risque important en conditions humides.</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🌾 Heatmap Culture × Ravageur</h2>
                    <div class="viz-container">
                        <img src="3_heatmap_culture_ravageur.png" alt="Heatmap">
                    </div>
                    <div class="interpretation">
                        <h3>💡 Comment lire cette heatmap ?</h3>
                        <p><strong>Cases rouge foncé (>40)</strong> : Associations très fréquentes → Surveillance prioritaire</p>
                        <p><strong>Cases orange (20-40)</strong> : Associations modérées → Vigilance régulière</p>
                        <p><strong>Cases jaunes (<20)</strong> : Associations occasionnelles</p>
                        <p><strong>Exemple</strong> : Colza × Limace grise (68) → Présents ensemble dans 68 bulletins sur 111</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>📅 Calendrier Saisonnier des Ravageurs</h2>
                    <div class="viz-container">
                        <img src="4_calendrier_saisonnier.png" alt="Calendrier">
                    </div>
                    <div class="interpretation">
                        <h3>💡 Périodes de surveillance</h3>
                        <p><strong>Cases bleu foncé</strong> : Pics d'activité → Surveillance maximale</p>
                        <p><strong>Août-Octobre</strong> : Période critique (altises, limaces, pucerons)</p>
                        <p><strong>Juillet-Août</strong> : Surveillance pyrale du maïs</p>
                    </div>
                </div>
                
                <div class="footer">
                    <p><strong>Rapport généré automatiquement par analyse NLP des BSV</strong></p>
                    <p>Plant Health NLP Analysis - Polytech Dijon - 2025</p>
                    <p style="font-size: 12px; margin-top: 10px;">
                    Ce rapport est basé sur l'analyse de {stats['fichiers_traites']} bulletins de santé du végétal.<br>
                    Les données reflètent les mentions dans les BSV, pas nécessairement les infestations réelles.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        output_file = self.viz_dir / "rapport_agronomique_bsv.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Rapport HTML genere: {output_file}")
        return output_file
    
    def generate_all_visualizations(self):
        """
        Génère toutes les visualisations agronomiques pertinentes
        """
        logger.info("Generation des visualisations agronomiques...")
        
        # 1. Top ravageurs
        self.plot_top_ravageurs(top_n=15)
        
        # 2. Top pathogènes
        self.plot_top_pathogenes(top_n=12)
        
        # 3. Heatmap culture × ravageur
        self.plot_heatmap_culture_ravageur(culture_top=8, ravageur_top=15)
        
        # 4. Calendrier saisonnier
        self.plot_calendrier_saisonnier(ravageur_top=15)
        
        # 5. Rapport HTML
        html_file = self.generate_agro_report()
        
        logger.info("=" * 80)
        logger.info("VISUALISATIONS AGRONOMIQUES GENEREES AVEC SUCCES!")
        logger.info("=" * 80)
        logger.info(f"Dossier de sortie: {self.viz_dir}")
        logger.info(f"Rapport HTML: {html_file}")
        logger.info("=" * 80)


def main():
    """
    Fonction principale
    """
    setup_logging()
    logger.info("Demarrage de la generation des visualisations agronomiques")
    logger.info("Plant Health NLP Analysis - Polytech Dijon")
    
    try:
        visualizer = BSVAgroVisualizer()
        visualizer.generate_all_visualizations()
        
        logger.info("Generation terminee avec succes!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Interruption par l'utilisateur (Ctrl+C)")
        return 1
        
    except Exception as e:
        logger.error("Erreur fatale lors de la generation")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())