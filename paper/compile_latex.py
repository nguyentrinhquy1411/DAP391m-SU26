import json
import requests
import sys

# Configure UTF-8 for console output on Windows to prevent UnicodeEncodeErrors
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

url = "https://latex.ytotech.com/builds/sync"
tex_path = "main.tex"
pdf_path = "maritime_sar.pdf"


def compile_latex():
    print(f"1. Reading LaTeX source code from: {tex_path}")
    try:
        with open(tex_path, 'r', encoding='utf-8') as f:
            latex_content = f.read()
    except Exception as e:
        print(f"   [Error] Failed to read LaTeX file: {e}")
        return

    payload = {
        "compiler": "pdflatex",
        "resources": [
            {
                "main": True,
                "content": latex_content
            }
        ]
    }
    
    # Check if figures/pipeline.png exists and add it to resources
    import os
    import base64
    if os.path.exists('figures/pipeline.png'):
        print("   [Info] Including figures/pipeline.png in compilation resources")
        with open('figures/pipeline.png', 'rb') as img_f:
            img_b64 = base64.b64encode(img_f.read()).decode('utf-8')
            payload["resources"].append({
                "path": "figures/pipeline.png",
                "file": img_b64
            })

    print(f"2. Sending request to compilation server: {url}")
    try:
        response = requests.post(url, json=payload, timeout=60)
    except Exception as e:
        print(f"   [Error] Failed to connect to compile API: {e}")
        return

    # Check for both 200 and 201 as successful build returns
    if response.status_code in [200, 201]:
        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' in content_type:
            print(f"3. Compilation successful (Status {response.status_code})! Writing PDF to: {pdf_path}")
            try:
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                print("   [Success] PDF compiled and saved successfully.")
            except Exception as e:
                print(f"   [Error] Failed to write PDF file: {e}")
        else:
            print(f"   [Warning] Received success status but unexpected Content-Type: {content_type}")
            # Safely print text fallback using replace
            print(response.text[:500].encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
    else:
        print(f"   [Error] Compilation server returned status code: {response.status_code}")
        print("   Server output details:")
        try:
            err_details = response.json()
            print(json.dumps(err_details, indent=2))
        except Exception:
            # Safely print text error using replace
            print(response.text[:1000].encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))

if __name__ == "__main__":
    compile_latex()
