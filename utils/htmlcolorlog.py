import io
import sys
import re
from contextlib import contextmanager

# ====================== ELEGANT LOGGING ENGINE ======================
class PipelineLogger:
    """
    A context manager that intercepts terminal output seamlessly.
    - Terminal: Sees raw output with colors.
    - In-Memory Buffer: Stores colored output to generate HTML.
    - Text File: Strips colors on the fly and saves pure text.

    We store the raw color data in memory (io.StringIO) and write the HTML directly from RAM. 
    No .tmp files, no WinError 32 crashes.
    """
    def __init__(self, text_path, html_path):
        self.original_stdout = sys.stdout
        self.text_path = text_path
        self.html_path = html_path
        self.ansi_buffer = io.StringIO() # Stores color data in RAM, no temp files!
        self.text_file = open(text_path, 'a', encoding='utf-8')
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, data):
        self.original_stdout.write(data)                    # 1. Output to real terminal
        self.ansi_buffer.write(data)                        # 2. Save to RAM for HTML
        clean_data = self.ansi_escape.sub('', data)         # 3. Strip colors
        self.text_file.write(clean_data)                    # 4. Save clean text to file
        self.flush()

    def flush(self):
        self.original_stdout.flush()
        self.text_file.flush()

    def isatty(self):
        # Trick third-party libraries into keeping colors enabled
        return True

    def __getattr__(self, attr):
        return getattr(self.original_stdout, attr)

    def generate_html_and_close(self):
        """Converts RAM buffer to HTML, then gracefully closes files."""
        try:
            from ansi2html import Ansi2HTMLConverter
            import re
            
            conv = Ansi2HTMLConverter(dark_bg=True, scheme='osx')
            raw_ansi = self.ansi_buffer.getvalue()
            
            # --- MINI TERMINAL SIMULATOR ---
            # Process terminal animations so HTML only sees the final static frame
            lines = raw_ansi.split('\n')
            final_lines = []
            
            for line in lines:
                # 1. Resolve Carriage Returns (\r): keep only text after the last \r
                if '\r' in line:
                    line = line.split('\r')[-1]
                    
                # 2. Count "Cursor Up" commands (\x1b[1A, \x1b[A) used in animations
                up_count = 0
                for match in re.finditer(r'\x1B\[([0-9]*)A', line):
                    val = match.group(1)
                    up_count += int(val) if val else 1
                    
                # 3. Simulate the cursor moving up by deleting previous lines in our buffer
                for _ in range(up_count):
                    if final_lines:
                        final_lines.pop()
                        
                # 4. Scrub remaining movement/clear codes so they don't corrupt the HTML
                line = re.sub(r'\x1B\[[0-9;]*[ABCDKJ]', '', line) # Scrub layout codes
                line = re.sub(r'\x1B\[\?[0-9]+[a-zA-Z]', '', line) # Scrub cursor hide/show
                
                final_lines.append(line)
                
            clean_ansi = '\n'.join(final_lines)
            # -------------------------------
            
            html = conv.convert(clean_ansi)
            
            with open(self.html_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"\n[INFO] 🌐 Colorful HTML Log saved to: {self.html_path}")
            
        except ImportError:
            print("\n[WARNING] 'ansi2html' not installed. HTML log skipped.")
        finally:
            self.ansi_buffer.close()
            self.text_file.close()

@contextmanager
def capture_and_log(text_path, html_path):
    """Context manager to safely wrap the execution block."""
    interceptor = PipelineLogger(text_path, html_path)
    sys.stdout = interceptor
    try:
        yield  # Run whatever is inside the 'with' block
    finally:
        sys.stdout = interceptor.original_stdout  # Safely restore terminal immediately
        interceptor.generate_html_and_close()     # Write HTML and cleanup