import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive.file']


# ====================================================================
# Authentification
# ====================================================================
def authenticate():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('drive', 'v3', credentials=creds)
    return service


# ====================================================================
# Vérifie ou crée un dossier sur Drive
# ====================================================================
def get_or_create_folder(service, folder_name, parent_id=None):
    """
    Retourne l'ID d'un dossier Drive existant ou le crée s'il n'existe pas.
    """
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get('files', [])
    if folders:
        return folders[0]['id']
    # Créer le dossier si inexistant
    metadata = {'name': folder_name, 'mimeType': 'application/vnd.google-apps.folder'}
    if parent_id:
        metadata['parents'] = [parent_id]
    folder = service.files().create(body=metadata, fields='id').execute()
    print(f"Dossier créé : {folder_name} -> id {folder.get('id')}")
    return folder.get('id')


# ====================================================================
# Upload conditionnel d'un fichier PDF
# ====================================================================
def upload_file(service, filepath, parent_id=None):
    filename = os.path.basename(filepath)
    query = f"name='{filename}'"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    existing_files = results.get('files', [])
    if existing_files:
        print(f"Le fichier '{filename}' existe déjà sur Drive. Ignoré.")
        return
    file_metadata = {'name': filename}
    if parent_id:
        file_metadata['parents'] = [parent_id]
    media = MediaFileUpload(filepath, mimetype='application/pdf', resumable=True)
    created = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"Upload: {filename} -> id: {created.get('id')}")


# ====================================================================
# Parcours récursif d'un dossier local pour recréer la structure sur Drive
# ====================================================================
def upload_folder_recursive(service, local_folder, parent_drive_id=None):
    """
    Parcourt un dossier local et ses sous-dossiers,
    crée les dossiers sur Drive et upload les PDFs correspondants.
    """
    folder_name = os.path.basename(local_folder)
    drive_folder_id = get_or_create_folder(service, folder_name, parent_drive_id)

    for entry in os.listdir(local_folder):
        path = os.path.join(local_folder, entry)
        if os.path.isdir(path):
            # Recurse dans le sous-dossier
            upload_folder_recursive(service, path, drive_folder_id)
        elif os.path.isfile(path) and entry.lower().endswith('.pdf'):
            # Upload PDF dans le dossier Drive correspondant
            upload_file(service, path, drive_folder_id)


# ====================================================================
# Partie principale
# ====================================================================
if __name__ == '__main__':

   

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # remonte d'un niveau depuis scripts/
    LOCAL_ROOT_FOLDER = os.path.join(BASE_DIR, 'data', 'raw', 'bourgogne_franche_comte')

    #LOCAL_ROOT_FOLDER = '..data/raw/bourgogne_franche_comte'  # dossier local racine à uploader

    if not os.path.isdir(LOCAL_ROOT_FOLDER):
        raise SystemExit(f"Le dossier local {LOCAL_ROOT_FOLDER} n'existe pas.")

    service = authenticate()

    # Upload récursif avec maintien de la structure
    upload_folder_recursive(service, LOCAL_ROOT_FOLDER)

    print("Terminé.")
