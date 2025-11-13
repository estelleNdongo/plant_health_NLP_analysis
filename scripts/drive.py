import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


class DriveUploader:
    """
    Classe pour synchroniser un dossier local avec Google Drive,
    en maintenant la structure des sous-dossiers et en évitant les doublons de PDF.
    """

    SCOPES = ['https://www.googleapis.com/auth/drive.file']

    def __init__(self, local_root: str, credentials_path: str = 'credentials.json', token_path: str = 'token.json'):
        """
        Initialise l'uploader Drive.
        :param local_root: Chemin absolu du dossier local à synchroniser.
        :param credentials_path: Chemin vers le fichier credentials.json (OAuth2).
        :param token_path: Chemin vers le fichier token.json (généré automatiquement après première authentification).
        """
        self.local_root = local_root
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = self.authenticate()

    # ====================================================================
    # Authentification
    # ====================================================================
    def authenticate(self):
        """
        Authentifie l’utilisateur Google et renvoie un service Drive prêt à l’emploi.
        """
        creds = None
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, self.SCOPES)

        # Rafraîchir ou créer de nouveaux identifiants
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(f"Le fichier {self.credentials_path} est introuvable.")
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, self.SCOPES)
                creds = flow.run_local_server(port=0)
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())

        # Construction du service Drive
        print("✅ Authentification réussie avec Google Drive.")
        return build('drive', 'v3', credentials=creds)

    # ====================================================================
    # Vérifie ou crée un dossier sur Drive
    # ====================================================================
    def get_or_create_folder(self, folder_name: str, parent_id: str = None) -> str:
        """
        Retourne l’ID du dossier Drive correspondant, ou le crée s’il n’existe pas.
        """
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        if parent_id:
            query += f" and '{parent_id}' in parents"

        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        folders = results.get('files', [])

        if folders:
            return folders[0]['id']

        # Création du dossier
        metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
        if parent_id:
            metadata['parents'] = [parent_id]

        folder = self.service.files().create(body=metadata, fields='id').execute()
        print(f"📁 Dossier créé : {folder_name} (id: {folder['id']})")
        return folder['id']

    # ====================================================================
    # Upload conditionnel d’un fichier PDF
    # ====================================================================
    def upload_file(self, filepath: str, parent_id: str = None):
        """
        Upload un fichier PDF sur Drive s’il n’existe pas déjà.
        """
        filename = os.path.basename(filepath)
        query = f"name='{filename}'"
        if parent_id:
            query += f" and '{parent_id}' in parents"

        # Vérifie existence
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        if results.get('files'):
            print(f"⚠️  Le fichier '{filename}' existe déjà. Ignoré.")
            return

        # Upload du fichier
        metadata = {'name': filename}
        if parent_id:
            metadata['parents'] = [parent_id]

        media = MediaFileUpload(filepath, mimetype='application/pdf', resumable=True)
        uploaded = self.service.files().create(body=metadata, media_body=media, fields='id').execute()
        print(f"✅ Upload réussi : {filename} (id: {uploaded['id']})")

    # ====================================================================
    # Upload récursif d’un dossier local
    # ====================================================================
    def upload_folder_recursive(self, local_folder: str, parent_drive_id: str = None):
        """
        Parcourt un dossier local et recrée la même arborescence sur Drive.
        """
        folder_name = os.path.basename(local_folder)
        drive_folder_id = self.get_or_create_folder(folder_name, parent_drive_id)

        for entry in os.listdir(local_folder):
            path = os.path.join(local_folder, entry)
            if os.path.isdir(path):
                self.upload_folder_recursive(path, drive_folder_id)
            elif os.path.isfile(path) and entry.lower().endswith('.pdf'):
                self.upload_file(path, drive_folder_id)

    # ====================================================================
    # Lancement global
    # ====================================================================
    def run(self):
        """
        Lance le processus complet d’upload.
        """
        if not os.path.isdir(self.local_root):
            raise SystemExit(f"❌ Le dossier local {self.local_root} n’existe pas.")
        print(f"🚀 Démarrage de l’upload depuis : {self.local_root}")
        self.upload_folder_recursive(self.local_root)
        print("🎉 Synchronisation terminée avec succès.")


# ====================================================================
# Script principal
# ====================================================================
if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOCAL_ROOT_FOLDER = os.path.join(BASE_DIR, 'data', 'raw', 'bourgogne_franche_comte')

    uploader = DriveUploader(local_root=LOCAL_ROOT_FOLDER)
    uploader.run()
