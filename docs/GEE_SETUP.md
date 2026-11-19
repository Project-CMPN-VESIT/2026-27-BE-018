# Enabling Google Earth Engine (optional)

STAC works with no setup. Enable GEE if you prefer Google's catalogue or STAC is blocked on your network.

## Option A: service account (recommended; works in Docker and unattended)

1. **Google Cloud project**: create or pick one at https://console.cloud.google.com and note its **project ID**.
2. **Enable the Earth Engine API** for the project (APIs & Services → Library → "Google Earth Engine API").
3. **Register the project for Earth Engine** at https://code.earthengine.google.com/register (choose the
   noncommercial or commercial use that applies to you).
4. **Service account**: IAM & Admin → Service Accounts → *Create*. Grant it the roles
   **Earth Engine Resource Viewer** and **Service Usage Consumer**.
5. **Key**: open the service account → Keys → *Add key* → JSON. Save the file as `secrets\gee-key.json`.
   Treat it as a password: it is git-ignored and mounted read-only in Docker.
6. **Configure** `.env`:

   ```ini
   GLN_GEE_ENABLED=true
   GLN_GEE_PROJECT=your-project-id
   GLN_GEE_SERVICE_ACCOUNT=name@your-project-id.iam.gserviceaccount.com
   # local run (Docker ignores this and uses /secrets/gee-key.json):
   GLN_GEE_KEY_FILE=D:\GLN\secrets\gee-key.json
   ```
7. **Verify**: `.\scripts\selfcheck.ps1 -Live -Provider gee`, then restart the app. "Google Earth Engine" becomes
   selectable under *Imagery → Source*. To make it the default, set `GLN_DEFAULT_PROVIDER=gee`.

## Option B: your own Google login (local only)

```powershell
backend\.venv\Scripts\earthengine.exe authenticate
```

Then set `GLN_GEE_ENABLED=true` and `GLN_GEE_PROJECT=...`, and leave `GLN_GEE_KEY_FILE` empty.

## Notes

* Collection: `COPERNICUS/S2_SR_HARMONIZED` (Sen2Cor L2A, baseline offset already removed → reflectance = DN/10000).
* Pixels are pulled with `computePixels` in 512 px blocks on the same UTM 10 m grid used for STAC, so results
  are comparable between providers.
* Earth Engine applies per-project quotas; very large areas or many parallel users can hit them.
