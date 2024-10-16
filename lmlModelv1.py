# Import necessary modules
import tkinter as tk
from tkinter import ttk
import threading
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import requests
from bs4 import BeautifulSoup
import numpy as np
import queue  # For thread-safe communication between threads

# Initialize a queue for thread-safe communication with the UI
ui_queue = queue.Queue()

# Step 1: Load BERT tokenizer and model
# If a saved model exists, load the model and tokenizer from the saved path
model_save_path = "saved_model"
if os.path.exists(model_save_path):
    tokenizer = BertTokenizer.from_pretrained(model_save_path)
    try:
        model = BertForSequenceClassification.from_pretrained(model_save_path)
    except Exception as e:
        print(f"Error loading model from {model_save_path}: {e}")
        # Load default pre-trained model if an error occurs while loading the saved model
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
else:
    # Load default pre-trained tokenizer and model if no saved model exists
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Move model to the appropriate device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Initialize DataLoaders globally to hold training and validation data
train_loader = None
val_loader = None

# Function Definitions

def update_scraping_status(source):
    """
    Update the scraping status in the UI from a separate thread.
    """
    ui_queue.put((update_scraping_status_ui, (source,)))

def update_scraping_status_ui(source):
    """
    Update the scraping status label with the current source being scraped.
    """
    if scraping_status_label.winfo_exists():
        try:
            scraping_status_label.config(text=f"Scraping from: {source}")
        except tk.TclError:
            pass  # Prevent updates if the label no longer exists
        scraping_status_label.update_idletasks()

def update_learning_status(text):
    """
    Add a text message to the learning status text box.
    """
    ui_queue.put((update_learning_status_ui, (text,)))

def update_learning_status_ui(text):
    """
    Insert new learning status information into the text box and scroll to the latest entry.
    """
    learning_status_text.insert(tk.END, text + "\n")
    learning_status_text.see(tk.END)
    learning_status_text.update_idletasks()

def update_output_label(text):
    """
    Update the output status label in the UI.
    """
    ui_queue.put((update_output_label_ui, (text,)))

def update_output_label_ui(text):
    """
    Update the status label in the UI with the given text.
    """
    if status_label.winfo_exists():
        try:
            status_label.config(text=text)
        except tk.TclError:
            pass  # Prevent updates if the label no longer exists
        status_label.update_idletasks()

def update_loss_graph(loss_values, val_loss_values):
    """
    Update the loss graph UI with new training and validation loss values.
    """
    ui_queue.put((update_loss_graph_ui, (loss_values.copy(), val_loss_values.copy())))

def update_loss_graph_ui(loss_values, val_loss_values):
    """
    Redraw the loss graph with the provided training and validation loss values.
    """
    loss_ax.clear()
    loss_ax.plot(range(1, len(loss_values) + 1), loss_values, marker='o', color='b', label='Training Loss')
    loss_ax.plot(range(1, len(val_loss_values) + 1), val_loss_values, marker='o', color='r', label='Validation Loss')
    loss_ax.set_title("Training and Validation Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.legend()
    loss_canvas.draw()

def update_progress_bar(progress):
    """
    Update the progress bar with the current progress value.
    """
    ui_queue.put((update_progress_bar_ui, (progress,)))

def update_progress_bar_ui(progress):
    """
    Update the progress bar in the UI.
    """
    progress_bar['value'] = progress
    progress_bar.update_idletasks()

def scrape_yahoo_finance():
    """
    Scrape headlines from Yahoo Finance homepage.
    """
    update_scraping_status("Yahoo Finance")
    url = "https://finance.yahoo.com/"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return []
    soup = BeautifulSoup(response.text, 'html.parser')
    texts = []
    for headline in soup.find_all('h3'):
        text = headline.get_text(strip=True)
        texts.append(text)
        update_learning_status(f"Scraped text: {text}")
    return texts

def scrape_investing():
    """
    Scrape headlines from Investing.com news page.
    """
    update_scraping_status("Investing.com")
    url = "https://www.investing.com/news/"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return []
    soup = BeautifulSoup(response.text, 'html.parser')
    texts = []
    for headline in soup.find_all('a', class_='title'):
        text = headline.get_text(strip=True)
        texts.append(text)
        update_learning_status(f"Scraped text: {text}")
    return texts

def scrape_coingecko():
    """
    Scrape news headlines from CoinGecko.
    """
    update_scraping_status("CoinGecko")
    url = "https://www.coingecko.com/en/news"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return []
    soup = BeautifulSoup(response.text, 'html.parser')
    texts = []
    for headline in soup.find_all('a', class_='tw-text-lg tw-no-underline'):
        text = headline.get_text(strip=True)
        texts.append(text)
        update_learning_status(f"Scraped text: {text}")
    return texts

def scrape_data(selected_sources):
    """
    Aggregate data from selected sources.
    """
    texts = []
    labels = []

    # Scrape from selected sources and assign random labels for demonstration purposes
    if "Yahoo Finance" in selected_sources:
        yahoo_texts = scrape_yahoo_finance()
        texts.extend(yahoo_texts)
        labels.extend(np.random.randint(0, 2, size=len(yahoo_texts)).tolist())

    if "Investing.com" in selected_sources:
        investing_texts = scrape_investing()
        texts.extend(investing_texts)
        labels.extend(np.random.randint(0, 2, size=len(investing_texts)).tolist())

    if "CoinGecko" in selected_sources:
        coingecko_texts = scrape_coingecko()
        texts.extend(coingecko_texts)
        labels.extend(np.random.randint(0, 2, size=len(coingecko_texts)).tolist())

    return pd.DataFrame({'text': texts, 'label': labels})

class SentimentDataset(Dataset):
    """
    A PyTorch Dataset class for preparing sentiment data for training.
    """
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get the tokenized input and corresponding label for a given index.
        """
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add special tokens (e.g., [CLS], [SEP])
            max_length=self.max_length,  # Truncate or pad to max_length
            return_token_type_ids=False,  # Not needed for classification
            padding='max_length',  # Pad sequences to the max length
            return_attention_mask=True,  # Return attention mask
            return_tensors='pt'  # Return PyTorch tensors
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten().to(device),  # Move to device
            'attention_mask': encoding['attention_mask'].flatten().to(device),  # Move to device
            'labels': torch.tensor(label, dtype=torch.long).to(device)  # Convert label to tensor and move to device
        }

def evaluate_model():
    """
    Evaluate the model on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if not torch.isnan(loss):
                total_loss += loss.item()
    # Calculate average validation loss
    avg_val_loss = total_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
    return avg_val_loss

def train_model(num_epochs, batch_size, learning_rate):
    """
    Train the BERT model with the given parameters.
    """
    model.train()  # Set the model to training mode
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Use Adam optimizer
    loss_values = []  # Track loss values for plotting
    val_loss_values = []  # Track validation loss values

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids']  # Input token IDs
            attention_mask = batch['attention_mask']  # Attention mask
            labels = batch['labels']  # True labels

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss  # Compute loss

            # Backward pass
            optimizer.zero_grad()  # Zero the gradients
            try:
                with torch.autograd.set_detect_anomaly(True):
                    loss.backward()  # Backpropagate the loss
            except RuntimeError as e:
                print(f"Error during backpropagation: {e}")
                continue  # Skip this batch if error occurs
            optimizer.step()  # Update the model parameters

            total_loss += loss.item()  # Accumulate the loss

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        loss_values.append(avg_loss)  # Append the average loss for plotting

        # Evaluate the model on the validation set
        avg_val_loss = evaluate_model()
        val_loss_values.append(avg_val_loss)  # Append validation loss for plotting

        # Update the UI label with training progress
        status_text = f"Epoch: {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
        update_output_label(status_text)

        # Update the loss graph
        update_loss_graph(loss_values, val_loss_values)

        # Update progress bar
        progress = int(100 * (epoch + 1) / num_epochs)
        update_progress_bar(progress)

    # Save the model and tokenizer
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Training complete
    update_output_label("Training complete.")

def start_training():
    """
    Start the training process by scraping data and training the model.
    """
    # Get training parameters from the UI
    num_epochs = int(epochs_entry.get())
    batch_size = int(batch_size_entry.get()) if batch_size_entry.get().isdigit() else 8
    learning_rate = float(learning_rate_entry.get())
    selected_sources = [source for source in scrape_sources if scrape_sources[source].get() == 1]

    if not selected_sources:
        update_output_label("Error: No data sources selected.")
        return

    # Scrape data based on selected sources
    def scrape_and_train():
        data = scrape_data(selected_sources)

        # Split data into train and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            data['text'].tolist(),  # List of text samples
            data['label'].tolist(),  # List of labels
            test_size=0.2,  # 20% of data for validation
            shuffle=True,
            random_state=42
        ) if len(data) > 0 else ([], [], [], [])

        if len(train_texts) == 0 or len(val_texts) == 0:
            update_output_label("Error: Not enough data to split into training and validation sets.")
            return

        # Create DataLoaders for training and validation data
        global train_loader, val_loader
        train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if len(train_texts) > 0 else None
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if len(val_texts) > 0 else None

        # Start the training process
        train_model(num_epochs, batch_size, learning_rate)

    # Start scraping and training in a separate thread to keep UI responsive
    threading.Thread(target=scrape_and_train).start()

# Create the main Tkinter window
window = tk.Tk()
window.title("LML Model Trainer")

# Configure the window layout
window.state('zoomed')  # Make the window full screen
window.resizable(True, True)  # Allow window resizing

# Training parameters UI components
tk.Label(window, text="Number of Epochs:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
epochs_entry = tk.Entry(window)
epochs_entry.grid(row=0, column=1, padx=10, pady=10)
epochs_entry.insert(0, "10")  # Default value for epochs

tk.Label(window, text="Batch Size:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
batch_size_entry = tk.Entry(window)
batch_size_entry.grid(row=1, column=1, padx=10, pady=10)
batch_size_entry.insert(0, "8")  # Default value for batch size

tk.Label(window, text="Learning Rate:").grid(row=2, column=0, padx=10, pady=10, sticky="e")
learning_rate_entry = tk.Entry(window)
learning_rate_entry.grid(row=2, column=1, padx=10, pady=10)
learning_rate_entry.insert(0, "0.001")  # Default value for learning rate

# Data sources selection UI components
tk.Label(window, text="Select Data Sources:").grid(row=3, column=0, padx=10, pady=10, sticky="e")
scrape_sources = {
    "Yahoo Finance": tk.IntVar(),
    "Investing.com": tk.IntVar(),
    "CoinGecko": tk.IntVar()
}
row_offset = 4
for i, (source, var) in enumerate(scrape_sources.items()):
    tk.Checkbutton(window, text=source, variable=var).grid(row=row_offset + i, column=1, sticky="w")

# Scraping Status Label
scraping_status_label = tk.Label(window, text="Scraping Status: Not started", relief="sunken", anchor="w")
scraping_status_label.grid(row=row_offset + len(scrape_sources) + 1, column=0, columnspan=2, padx=10, pady=10, sticky="we")

# Real-time Data Learning Status
learning_status_text = tk.Text(window, height=10, width=70, wrap='word', relief="sunken")
learning_status_text.grid(row=row_offset + len(scrape_sources) + 2, column=0, columnspan=2, padx=10, pady=10, sticky="we")
learning_status_text.insert('1.0', "Learning status updates will appear here...\n")

# Status Label
status_label = tk.Label(window, text="Status: Waiting to start...", relief="sunken", anchor="w")
status_label.grid(row=row_offset + len(scrape_sources) + 3, column=0, columnspan=2, padx=10, pady=10, sticky="we")

# Start Training Button
start_button = tk.Button(window, text="Start Training", command=lambda: threading.Thread(target=start_training).start())
start_button.grid(row=row_offset + len(scrape_sources) + 4, column=0, columnspan=2, padx=10, pady=20)

# Progress Bar
progress_bar = ttk.Progressbar(window, orient="horizontal", length=300, mode="determinate")
progress_bar.grid(row=row_offset + len(scrape_sources) + 5, column=0, columnspan=2, padx=10, pady=10)

# Loss Graph UI components
fig, loss_ax = plt.subplots(figsize=(10, 6))  # Set larger figure size for better visibility
loss_canvas = FigureCanvasTkAgg(fig, master=window)
loss_canvas.get_tk_widget().grid(row=row_offset + len(scrape_sources) + 6, column=0, columnspan=2, padx=20, pady=20, sticky='nsew')

# Start the Tkinter event loop
def process_queue():
    """
    Process UI updates from the queue to ensure thread-safe UI operations.
    """
    while not ui_queue.empty():
        try:
            func, args = ui_queue.get_nowait()
            func(*args)
        except queue.Empty:
            pass
    window.after(100, process_queue)  # Check the queue every 100 ms

window.after(100, process_queue)  # Start the queue processing
window.mainloop()  # Run the main event loop
