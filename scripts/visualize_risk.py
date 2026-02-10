import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

# Configuration du style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Couleurs par niveau de risque
RISK_COLORS = {
    'très faible': '#2ecc71',  # Vert
    'faible': '#95a5a6',        # Gris clair
    'modéré': '#f39c12',        # Orange
    'élevé': '#e74c3c',         # Rouge
    'très élevé': '#c0392b'     # Rouge foncé
}

RISK_ORDER = ['très faible', 'faible', 'modéré', 'élevé', 'très élevé']

MONTH_NAMES = {
    1: 'Jan', 2: 'Fév', 3: 'Mar', 4: 'Avr', 5: 'Mai', 6: 'Juin',
    7: 'Juil', 8: 'Août', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Déc'
}


class RiskVisualizer:
    """
    Génère les visualisations des risques détectés par NLP
    """
    
    def __init__(self):
        """Initialise le visualiseur"""
        self.config_loader = ConfigLoader("config.yaml")
        self.base_dir = Path(self.config_loader.base_dir)
        
        # Chemins
        self.results_dir = self.base_dir / self.config_loader.config["data"]["results_dir"]
        self.risk_dir = self.results_dir / "risk_analysis_nlp"
        self.viz_dir = self.results_dir / "visualizations_risk"
        
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Données
        self.all_risks = []
        self.report = None
        
        logger.info("Visualiseur de risques initialise")
    
    def load_data(self):
        """Charge toutes les données de risque"""
        logger.info("Chargement des donnees...")
        
        # Charger le rapport global
        report_path = self.risk_dir / "risk_analysis_nlp_report.json"
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                self.report = json.load(f)
            logger.info(f"Rapport global charge: {self.report['statistiques_globales']['mentions_risque_total']} mentions")
        else:
            logger.error(f"Rapport non trouve: {report_path}")
            return False
        
        # Charger tous les fichiers individuels
        for year_dir in sorted(self.risk_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            
            for json_file in year_dir.glob("*_risks_nlp.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for risk in data.get('risques_detectes', []):
                            risk['year'] = data['year']
                            risk['filename'] = data['filename']
                            self.all_risks.append(risk)
                except Exception as e:
                    logger.warning(f"Erreur chargement {json_file.name}: {e}")
        
        logger.info(f"Total risques charges: {len(self.all_risks)}")
        return True
    
    def plot_risk_distribution(self):
        """Graphique 1: Distribution des niveaux de risque (Pie + Bar)"""
        logger.info("Creation du graphique de distribution des risques...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Données
        niveaux = self.report['analyse_par_niveau']
        levels = []
        counts = []
        colors = []
        
        for level in RISK_ORDER:
            if level in niveaux:
                levels.append(level.capitalize())
                counts.append(niveaux[level]['occurrences'])
                colors.append(RISK_COLORS[level])
        
        # Pie chart
        ax1.pie(counts, labels=levels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Répartition des Niveaux de Risque\n(1219 mentions)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Bar chart
        bars = ax2.bar(levels, counts, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Niveau de Risque', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Nombre de Mentions', fontsize=12, fontweight='bold')
        ax2.set_title('Distribution des Risques par Niveau', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(axis='y', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        output_path = self.viz_dir / "1_distribution_risques.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graphique sauvegarde: {output_path}")
    
    def plot_top_entities(self):
        """Graphique 2: Top entités à risque"""
        logger.info("Creation du graphique top entites...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Top 15 entités
        top_entities = self.report['top_entites_a_risque'][:15]
        
        names = [e['entite'].title() for e in top_entities]
        scores = [e['score_moyen'] for e in top_entities]
        
        # Couleurs selon le score
        colors_bars = []
        for score in scores:
            if score >= 3:
                colors_bars.append(RISK_COLORS['très élevé'])
            elif score >= 2.5:
                colors_bars.append(RISK_COLORS['élevé'])
            elif score >= 2:
                colors_bars.append(RISK_COLORS['modéré'])
            elif score >= 1:
                colors_bars.append(RISK_COLORS['faible'])
            else:
                colors_bars.append(RISK_COLORS['très faible'])
        
        # Barres horizontales
        y_pos = np.arange(len(names))
        bars = ax.barh(y_pos, scores, color=colors_bars, edgecolor='black', linewidth=1.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Score Moyen de Risque (0-4)', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 Ravageurs et Pathogènes à Risque Élevé\n(Score moyen sur 3 ans)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 4)
        ax.grid(axis='x', alpha=0.3)
        
        # Ligne de référence à 2.5 (élevé)
        ax.axvline(x=2.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Seuil Élevé')
        ax.legend()
        
        # Ajouter les scores
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.05, i, f'{score:.2f}', 
                   va='center', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        output_path = self.viz_dir / "2_top_entites_risque.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graphique sauvegarde: {output_path}")
    
    def plot_heatmap_entity_risk(self):
        """Graphique 3: Heatmap Entité × Niveau de Risque"""
        logger.info("Creation de la heatmap entite x niveau...")
        
        # Top 12 entités
        top_entities = self.report['top_entites_a_risque'][:12]
        
        # Construire la matrice
        matrix = []
        entity_names = []
        
        for entity_data in top_entities:
            entity_names.append(entity_data['entite'].title())
            repartition = entity_data['repartition']
            total = entity_data['mentions_total']
            
            # Pourcentages pour chaque niveau
            row = []
            for level in RISK_ORDER:
                count = repartition.get(level, 0)
                percentage = (count / total * 100) if total > 0 else 0
                row.append(percentage)
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # Créer la heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
        
        # Axes
        ax.set_xticks(np.arange(len(RISK_ORDER)))
        ax.set_yticks(np.arange(len(entity_names)))
        ax.set_xticklabels([level.capitalize() for level in RISK_ORDER], fontsize=11)
        ax.set_yticklabels(entity_names, fontsize=10)
        
        # Rotation des labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Ajouter les valeurs
        for i in range(len(entity_names)):
            for j in range(len(RISK_ORDER)):
                text = ax.text(j, i, f'{matrix[i, j]:.0f}%',
                             ha="center", va="center", color="black" if matrix[i, j] < 50 else "white",
                             fontweight='bold', fontsize=9)
        
        ax.set_title('Distribution des Niveaux de Risque par Entité\n(% des mentions)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Pourcentage (%)', rotation=270, labelpad=20, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.viz_dir / "3_heatmap_entite_niveau.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graphique sauvegarde: {output_path}")
    
    def plot_temporal_evolution(self):
        """Graphique 4: Évolution temporelle du risque"""
        logger.info("Creation du graphique d evolution temporelle...")
        
        # Calculer le score moyen par mois
        monthly_scores = defaultdict(list)
        
        for risk in self.all_risks:
            mois = risk.get('mois')
            score = risk.get('score', 0)
            if mois:
                monthly_scores[mois].append(score)
        
        # Moyennes
        months = sorted(monthly_scores.keys())
        avg_scores = [np.mean(monthly_scores[m]) for m in months]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Courbe
        ax.plot(months, avg_scores, marker='o', linewidth=2.5, markersize=8, 
               color='#3498db', label='Score Moyen')
        
        # Zones de risque
        ax.axhspan(0, 1, alpha=0.1, color=RISK_COLORS['très faible'], label='Très Faible')
        ax.axhspan(1, 2, alpha=0.1, color=RISK_COLORS['faible'])
        ax.axhspan(2, 2.5, alpha=0.1, color=RISK_COLORS['modéré'], label='Modéré')
        ax.axhspan(2.5, 3.5, alpha=0.1, color=RISK_COLORS['élevé'], label='Élevé')
        ax.axhspan(3.5, 4, alpha=0.1, color=RISK_COLORS['très élevé'], label='Très Élevé')
        
        ax.set_xlabel('Mois', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score Moyen de Risque', fontsize=12, fontweight='bold')
        ax.set_title('Évolution du Risque Moyen au Fil de l\'Année\n(2022-2024)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 4)
        ax.set_xticks(months)
        ax.set_xticklabels([MONTH_NAMES.get(m, m) for m in months])
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        output_path = self.viz_dir / "4_evolution_temporelle.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graphique sauvegarde: {output_path}")
    
    def plot_seasonal_calendar(self):
        """Graphique 5: Calendrier saisonnier des risques par entité"""
        logger.info("Creation du calendrier saisonnier...")
        
        # Top 10 entités
        top_entities_list = [e['entite'] for e in self.report['top_entites_a_risque'][:10]]
        
        # Matrice entité × mois (score moyen)
        matrix = np.zeros((len(top_entities_list), 12))
        
        for risk in self.all_risks:
            mois = risk.get('mois')
            score = risk.get('score', 0)
            
            # Entités de ce risque
            entities = risk.get('ravageurs', []) + risk.get('pathogenes', [])
            
            if mois and 1 <= mois <= 12:
                for entity in entities:
                    if entity in top_entities_list:
                        idx = top_entities_list.index(entity)
                        # Accumuler pour moyenner ensuite
                        if matrix[idx, mois-1] == 0:
                            matrix[idx, mois-1] = score
                        else:
                            matrix[idx, mois-1] = (matrix[idx, mois-1] + score) / 2
        
        # Créer la heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=4)
        
        # Axes
        ax.set_xticks(np.arange(12))
        ax.set_yticks(np.arange(len(top_entities_list)))
        ax.set_xticklabels([MONTH_NAMES[i+1] for i in range(12)], fontsize=11)
        ax.set_yticklabels([e.title() for e in top_entities_list], fontsize=10)
        
        ax.set_title('Calendrier Saisonnier des Risques\n(Score moyen par mois et par entité)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score de Risque Moyen', rotation=270, labelpad=20, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.viz_dir / "5_calendrier_saisonnier.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graphique sauvegarde: {output_path}")
    
    def generate_html_report(self):
        """Génère un rapport HTML interactif"""
        logger.info("Generation du rapport HTML...")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport d'Analyse des Risques - BSV Bourgogne-Franche-Comté</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #3498db;
            margin: 10px 0;
        }}
        
        .stat-label {{
            font-size: 1.1em;
            color: #7f8c8d;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section h2 {{
            font-size: 2em;
            color: #2c3e50;
            margin-bottom: 20px;
            border-left: 5px solid #3498db;
            padding-left: 15px;
        }}
        
        .visualization {{
            margin: 30px 0;
            text-align: center;
        }}
        
        .visualization img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .visualization h3 {{
            margin: 20px 0 10px 0;
            color: #34495e;
            font-size: 1.5em;
        }}
        
        .visualization p {{
            color: #7f8c8d;
            line-height: 1.6;
            max-width: 800px;
            margin: 10px auto;
        }}
        
        .insights {{
            background: #ecf0f1;
            padding: 30px;
            border-radius: 10px;
            margin: 30px 0;
        }}
        
        .insights h3 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.4em;
        }}
        
        .insights ul {{
            list-style: none;
        }}
        
        .insights li {{
            padding: 10px 0;
            border-bottom: 1px solid #bdc3c7;
            color: #34495e;
            line-height: 1.6;
        }}
        
        .insights li:last-child {{
            border-bottom: none;
        }}
        
        .insights li::before {{
            content: "✓ ";
            color: #27ae60;
            font-weight: bold;
            margin-right: 10px;
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .footer p {{
            opacity: 0.8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Analyse des Risques Phytosanitaires</h1>
            <p>BSV Grandes Cultures - Bourgogne-Franche-Comté (2022-2024)</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Méthode : NLP Pur (Syntaxe + Embeddings)</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Fichiers Analysés</div>
                <div class="stat-value">{self.report['statistiques_globales']['fichiers_analyses']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mentions de Risque</div>
                <div class="stat-value">{self.report['statistiques_globales']['mentions_risque_total']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Entités Détectées</div>
                <div class="stat-value">{len(self.report['top_entites_a_risque'])}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Période</div>
                <div class="stat-value">3 ans</div>
            </div>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>🎯 Vue d'Ensemble</h2>
                <div class="insights">
                    <h3>Points Clés de l'Analyse</h3>
                    <ul>
                        <li>Plus de {self.report['statistiques_globales']['mentions_risque_total']} mentions de risque détectées automatiquement sur 3 ans</li>
                        <li>Détection basée sur l'analyse syntaxique et les word embeddings (IA)</li>
                        <li>{len(self.report['top_entites_a_risque'])} ravageurs et pathogènes identifiés à risque</li>
                        <li>Analyse de {self.report['statistiques_globales']['fichiers_analyses']} bulletins BSV</li>
                        <li>Couverture temporelle complète des campagnes 2022, 2023 et 2024</li>
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Distribution des Risques</h2>
                <div class="visualization">
                    <img src="1_distribution_risques.png" alt="Distribution des risques">
                    <h3>Répartition par Niveau de Risque</h3>
                    <p>
                        Cette visualisation montre la distribution globale des {self.report['statistiques_globales']['mentions_risque_total']} mentions de risque. 
                        On observe que {self.report['analyse_par_niveau'].get('élevé', {}).get('pourcentage', 0) + self.report['analyse_par_niveau'].get('très élevé', {}).get('pourcentage', 0):.1f}% 
                        des risques sont classés comme élevés ou très élevés, nécessitant une surveillance renforcée.
                    </p>
                </div>
            </div>
            
            <div class="section">
                <h2>🐛 Ravageurs et Pathogènes Prioritaires</h2>
                <div class="visualization">
                    <img src="2_top_entites_risque.png" alt="Top entités à risque">
                    <h3>Top 15 des Entités à Risque Élevé</h3>
                    <p>
                        Ce graphique classe les ravageurs et pathogènes selon leur score moyen de risque sur 3 ans. 
                        Les entités avec un score supérieur à 2.5 nécessitent une attention particulière et une surveillance continue.
                    </p>
                </div>
                <div class="insights">
                    <h3>Top 5 Entités à Surveiller en Priorité</h3>
                    <ul>
"""
        
        # Ajouter le top 5
        for i, entity in enumerate(self.report['top_entites_a_risque'][:5], 1):
            html_content += f"                        <li><strong>{entity['entite'].title()}</strong> : Score moyen {entity['score_moyen']:.2f}/4 ({entity['mentions_total']} mentions)</li>\n"
        
        html_content += """
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2>🔥 Profils de Risque par Entité</h2>
                <div class="visualization">
                    <img src="3_heatmap_entite_niveau.png" alt="Heatmap entité × niveau">
                    <h3>Distribution des Niveaux de Risque</h3>
                    <p>
                        Cette heatmap montre la répartition en pourcentage des différents niveaux de risque pour chaque entité. 
                        Les cellules rouges indiquent un pourcentage élevé de mentions à risque élevé pour cette entité.
                    </p>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Évolution Temporelle</h2>
                <div class="visualization">
                    <img src="4_evolution_temporelle.png" alt="Évolution temporelle">
                    <h3>Risque Moyen par Mois</h3>
                    <p>
                        Cette courbe montre l'évolution du score moyen de risque au fil des mois. 
                        Les zones colorées représentent les différents niveaux de risque pour faciliter l'interprétation.
                    </p>
                </div>
            </div>
            
            <div class="section">
                <h2>📅 Calendrier Saisonnier</h2>
                <div class="visualization">
                    <img src="5_calendrier_saisonnier.png" alt="Calendrier saisonnier">
                    <h3>Périodes de Risque par Entité</h3>
                    <p>
                        Ce calendrier indique quand chaque ravageur/pathogène présente le plus de risque au cours de l'année. 
                        Utilisez cette visualisation pour planifier vos observations et interventions.
                    </p>
                </div>
            </div>
            
            <div class="section">
                <h2>💡 Méthodologie</h2>
                <div class="insights">
                    <h3>Approche NLP Pure (Intelligence Artificielle)</h3>
                    <ul>
                        <li><strong>Analyse Syntaxique</strong> : Détection automatique des patterns "risque est élevé", "risque devient important", etc.</li>
                        <li><strong>Word Embeddings</strong> : Classification des adjectifs par similarité sémantique (IA)</li>
                        <li><strong>Gestion des Négations</strong> : Détection automatique de "pas de risque" → très faible</li>
                        <li><strong>Matching Intelligent</strong> : Association automatique des entités (ravageurs/pathogènes) aux mentions de risque</li>
                        <li><strong>Confiance Quantifiée</strong> : Chaque détection a un score de confiance (0-1)</li>
                        <li><strong>Sans Dictionnaire Manuel</strong> : Le système apprend automatiquement à classifier les niveaux de risque</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}</p>
            <p>Plant Health NLP Analysis - Polytech Dijon</p>
            <p style="margin-top: 10px; font-size: 0.9em;">Méthode : NLP Pur (Syntaxe + Embeddings) | {self.report['statistiques_globales']['mentions_risque_total']} risques analysés</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Sauvegarder le HTML
        html_path = self.viz_dir / "rapport_analyse_risques.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Rapport HTML sauvegarde: {html_path}")
    
    def generate_all(self):
        """Génère toutes les visualisations"""
        logger.info("=" * 80)
        logger.info("GENERATION DES VISUALISATIONS")
        logger.info("=" * 80)
        
        if not self.load_data():
            logger.error("Impossible de charger les donnees")
            return False
        
        # Générer les graphiques
        self.plot_risk_distribution()
        self.plot_top_entities()
        self.plot_heatmap_entity_risk()
        self.plot_temporal_evolution()
        self.plot_seasonal_calendar()
        
        # Générer le rapport HTML
        self.generate_html_report()
        
        logger.info("=" * 80)
        logger.info("VISUALISATIONS TERMINEES")
        logger.info(f"Dossier: {self.viz_dir}")
        logger.info("=" * 80)
        
        return True


def main():
    """Fonction principale"""
    setup_logging()
    logger.info("Demarrage de la generation des visualisations")
    logger.info("Plant Health NLP Analysis - Polytech Dijon")
    
    try:
        visualizer = RiskVisualizer()
        visualizer.generate_all()
        
        logger.info("Generation terminee avec succes!")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Interruption par l'utilisateur (Ctrl+C)")
        return 1
        
    except Exception as e:
        logger.error("Erreur fatale")
        logger.exception(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())