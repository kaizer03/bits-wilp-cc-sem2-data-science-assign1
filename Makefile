# ─────────────────────────────────────────────────────────────────────────────
# Mobile Price Prediction — Project Commands
# BITS Pilani WILP · Introduction to Data Science · Assignment 1
# ─────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   make <target>
#
# All commands use the virtual environment at mobile-price-ml/venv/ directly,
# so you do NOT need to activate the venv before running them.
# ─────────────────────────────────────────────────────────────────────────────

VENV        := mobile-price-ml/venv
PYTHON      := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
STREAMLIT   := $(VENV)/bin/streamlit
JUPYTER     := $(VENV)/bin/jupyter
WORKDIR     := mobile-price-ml

DEPS        := numpy pandas scikit-learn matplotlib seaborn streamlit joblib notebook
NOTEBOOK_LOG  := mobile-price-ml/jupyter.log
NOTEBOOK_PID  := mobile-price-ml/.jupyter.pid
ADVANCED_LOG  := mobile-price-ml/advanced-ui.log
ADVANCED_PID  := mobile-price-ml/.advanced-ui.pid
SIMPLE_LOG    := mobile-price-ml/simple-ui.log
SIMPLE_PID    := mobile-price-ml/.simple-ui.pid

.PHONY: help venv install setup \
        notebook stop-notebook \
        advanced-ui-prediction stop-advanced-ui-prediction \
        simple-ui-prediction stop-simple-ui-prediction \
        export-pdf activate

# Default target — print available commands
help:
	@echo ""
	@echo "  Mobile Price Prediction — Available Commands"
	@echo "  ──────────────────────────────────────────────────────────"
	@echo "  make setup      Create venv and install all dependencies"
	@echo "  make venv       Create the Python virtual environment only"
	@echo "  make install    Install all required packages into the venv"
	@echo "  make notebook       Launch Jupyter Notebook in background (Group 13.ipynb)"
	@echo "  make stop-notebook  Stop the background Jupyter Notebook server"
	@echo "  make advanced-ui-prediction       Run advanced-ui-prediction.py in background via Streamlit"
	@echo "  make stop-advanced-ui-prediction  Stop the background advanced Streamlit app"
	@echo "  make simple-ui-prediction         Run simple-ui-prediction.py  in background via Streamlit"
	@echo "  make stop-simple-ui-prediction    Stop the background simple Streamlit app"
	@echo "  make export-pdf     Export notebook to HTML (open in browser → Print → Save as PDF)"
	@echo "  make activate       Print the command to activate the venv manually"
	@echo "  ──────────────────────────────────────────────────────────"
	@echo ""

# Create the virtual environment if it doesn't already exist
venv:
	@if [ -d "$(VENV)" ]; then \
		echo "Virtual environment already exists at $(VENV)"; \
	else \
		echo "Creating virtual environment at $(VENV)..."; \
		python3 -m venv $(VENV); \
		echo "Done."; \
	fi

# Install all required dependencies into the venv
install: venv
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install $(DEPS)
	@echo "All dependencies installed."

# Create venv and install dependencies in one step
setup: venv install
	@echo ""
	@echo "Setup complete. Run 'make notebook', 'make advanced-ui-prediction', or 'make simple-ui-prediction' to get started."

# Launch Jupyter Notebook in the background — frees the terminal immediately
notebook:
	@if [ -f "$(NOTEBOOK_PID)" ] && kill -0 $$(cat "$(NOTEBOOK_PID)") 2>/dev/null; then \
		echo "Jupyter Notebook is already running (PID $$(cat $(NOTEBOOK_PID)))."; \
		echo "Run 'make stop-notebook' first, or open http://localhost:8888 in your browser."; \
	else \
		echo "Starting Jupyter Notebook in the background..."; \
		nohup sh -c 'cd $(WORKDIR) && exec ../$(JUPYTER) notebook "Group 13.ipynb" --no-browser --port=8888' \
			> $(NOTEBOOK_LOG) 2>&1 & echo $$! > $(NOTEBOOK_PID); \
		sleep 4; \
		echo "Jupyter Notebook started (PID $$(cat $(NOTEBOOK_PID)))."; \
		echo ""; \
		echo "  Open URL (with token) — copy this into any browser:"; \
		echo "  $$(grep -o 'http://localhost:[0-9]*/[^ ]*token=[^ ]*' $(NOTEBOOK_LOG) | head -1)"; \
		echo ""; \
		echo "Log file:  $(NOTEBOOK_LOG)"; \
		echo "Stop with: make stop-notebook"; \
	fi

# Stop the background Jupyter Notebook server
stop-notebook:
	@if [ -f "$(NOTEBOOK_PID)" ] && kill -0 $$(cat "$(NOTEBOOK_PID)") 2>/dev/null; then \
		echo "Stopping Jupyter Notebook (PID $$(cat $(NOTEBOOK_PID)))..."; \
		kill $$(cat "$(NOTEBOOK_PID)"); \
		rm -f "$(NOTEBOOK_PID)"; \
		echo "Jupyter Notebook stopped."; \
	else \
		echo "No running Jupyter Notebook found."; \
		rm -f "$(NOTEBOOK_PID)"; \
	fi

# Run advanced-ui-prediction.py in the background via Streamlit
advanced-ui-prediction:
	@if [ -f "$(ADVANCED_PID)" ] && kill -0 $$(cat "$(ADVANCED_PID)") 2>/dev/null; then \
		echo "Advanced UI is already running (PID $$(cat $(ADVANCED_PID)))."; \
		echo "Run 'make stop-advanced-ui-prediction' first, or open http://localhost:8501 in your browser."; \
	else \
		echo "Starting Streamlit → advanced-ui-prediction.py in the background..."; \
		nohup sh -c 'cd $(WORKDIR) && exec ../$(STREAMLIT) run advanced-ui-prediction.py --server.port 8501 --server.headless true' \
			> $(ADVANCED_LOG) 2>&1 & echo $$! > $(ADVANCED_PID); \
		sleep 2; \
		echo "Advanced UI started (PID $$(cat $(ADVANCED_PID)))."; \
		echo "Open your browser at:  http://localhost:8501"; \
		echo "Log file:              $(ADVANCED_LOG)"; \
		echo "Stop with:             make stop-advanced-ui-prediction"; \
	fi

# Stop the background advanced Streamlit app
stop-advanced-ui-prediction:
	@if [ -f "$(ADVANCED_PID)" ] && kill -0 $$(cat "$(ADVANCED_PID)") 2>/dev/null; then \
		echo "Stopping advanced-ui-prediction.py (PID $$(cat $(ADVANCED_PID)))..."; \
		kill $$(cat "$(ADVANCED_PID)"); \
		rm -f "$(ADVANCED_PID)"; \
		echo "Advanced UI stopped."; \
	else \
		echo "No running advanced UI found."; \
		rm -f "$(ADVANCED_PID)"; \
	fi

# Run simple-ui-prediction.py in the background via Streamlit
simple-ui-prediction:
	@if [ -f "$(SIMPLE_PID)" ] && kill -0 $$(cat "$(SIMPLE_PID)") 2>/dev/null; then \
		echo "Simple UI is already running (PID $$(cat $(SIMPLE_PID)))."; \
		echo "Run 'make stop-simple-ui-prediction' first, or open http://localhost:8502 in your browser."; \
	else \
		echo "Starting Streamlit → simple-ui-prediction.py in the background..."; \
		nohup sh -c 'cd $(WORKDIR) && exec ../$(STREAMLIT) run simple-ui-prediction.py --server.port 8502 --server.headless true' \
			> $(SIMPLE_LOG) 2>&1 & echo $$! > $(SIMPLE_PID); \
		sleep 2; \
		echo "Simple UI started (PID $$(cat $(SIMPLE_PID)))."; \
		echo "Open your browser at:  http://localhost:8502"; \
		echo "Log file:              $(SIMPLE_LOG)"; \
		echo "Stop with:             make stop-simple-ui-prediction"; \
	fi

# Stop the background simple Streamlit app
stop-simple-ui-prediction:
	@if [ -f "$(SIMPLE_PID)" ] && kill -0 $$(cat "$(SIMPLE_PID)") 2>/dev/null; then \
		echo "Stopping simple-ui-prediction.py (PID $$(cat $(SIMPLE_PID)))..."; \
		kill $$(cat "$(SIMPLE_PID)"); \
		rm -f "$(SIMPLE_PID)"; \
		echo "Simple UI stopped."; \
	else \
		echo "No running simple UI found."; \
		rm -f "$(SIMPLE_PID)"; \
	fi

# Export notebook to HTML — bypasses XeLaTeX entirely, avoiding PDF conversion errors
# Open the generated HTML in your browser and use File > Print > Save as PDF
export-pdf:
	@echo "Exporting notebook to HTML (bypasses LaTeX PDF pipeline)..."
	cd $(WORKDIR) && ../$(JUPYTER) nbconvert "Group 13.ipynb" --to html \
		--output "Group 13.html"
	@echo ""
	@echo "Done. HTML export written to: mobile-price-ml/Group 13.html"
	@echo "Open the file in your browser, then use File > Print > Save as PDF."
	@echo ""

# Print the manual venv activation command
activate:
	@echo ""
	@echo "  To activate the virtual environment in your current shell, run:"
	@echo ""
	@echo "    source $(VENV)/bin/activate"
	@echo ""
	@echo "  To deactivate later, run:  deactivate"
	@echo ""
