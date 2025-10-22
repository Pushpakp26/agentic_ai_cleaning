class DataCleanerApp {
    constructor() {
        this.currentFile = null;
        this.eventSource = null;
        // Use absolute URL for API calls to support both file:// and http:// access
        this.apiBaseUrl = window.location.protocol === 'file:' 
            ? 'http://localhost:8000' 
            : '';
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // File input
        const fileInput = document.getElementById('file-input');
        const uploadArea = document.getElementById('upload-area');

        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        uploadArea.addEventListener('click', () => fileInput.click());

        // Buttons
        document.getElementById('start-processing').addEventListener('click', () => this.startProcessing());
        document.getElementById('remove-file').addEventListener('click', () => this.removeFile());
        document.getElementById('process-another').addEventListener('click', () => this.resetApp());
        document.getElementById('retry-processing').addEventListener('click', () => this.resetApp());
        document.getElementById('download-dataset').addEventListener('click', () => this.downloadFile('dataset'));
        document.getElementById('download-report').addEventListener('click', () => this.downloadFile('report'));
    }

    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
        // Validate file type
        const allowedTypes = ['text/csv', 'application/json', 'application/octet-stream'];
        const allowedExtensions = ['.csv', '.json', '.parquet'];

        const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

        if (!allowedExtensions.includes(fileExtension)) {
            this.showError('Please select a CSV, JSON, or Parquet file.');
            return;
        }

        // Validate file size (1GB limit)
        const maxSize = 1024 * 1024 * 1024; // 1GB
        if (file.size > maxSize) {
            this.showError('File size must be less than 1GB.');
            return;
        }

        this.currentFile = file;
        this.displayFileInfo(file);
    }

    displayFileInfo(file) {
        document.getElementById('file-name').textContent = file.name;
        document.getElementById('file-size').textContent = this.formatFileSize(file.size);
        document.getElementById('file-info').classList.remove('hidden');
        document.getElementById('upload-area').classList.add('hidden');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    removeFile() {
        this.currentFile = null;
        document.getElementById('file-info').classList.add('hidden');
        document.getElementById('upload-area').classList.remove('hidden');
        document.getElementById('file-input').value = '';
    }

    async startProcessing() {
        if (!this.currentFile) {
            this.showError('Please select a file first.');
            return;
        }

        this.showLoading(true);
        this.switchSection('processing-section');
        this.initializeProgress();

        try {
            // Upload file
            const uploadResponse = await this.uploadFile(this.currentFile);

            if (!uploadResponse.ok) {
                const errorData = await uploadResponse.json();
                throw new Error(errorData.detail || 'Upload failed');
            }

            const uploadData = await uploadResponse.json();
            this.logMessage(`File uploaded successfully: ${uploadData.filename}`, 'success');

            // Start processing
            await this.startProcessingPipeline(uploadData.filename);

        } catch (error) {
            console.error('Processing error:', error);
            this.showError(error.message);
            this.showLoading(false);
        }
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        return fetch(`${this.apiBaseUrl}/api/upload/`, {
            method: 'POST',
            body: formData
        });
    }

    async startProcessingPipeline(filename) {
        try {
            // Start processing and listen to SSE
            this.eventSource = new EventSource(`${this.apiBaseUrl}/api/process/start/${encodeURIComponent(filename)}`);
            this.processingComplete = false;

            this.eventSource.onmessage = (event) => {
                this.handleProgressEvent(event);
            };

            this.eventSource.onerror = (error) => {
                console.error('SSE error:', error);
                if (!this.processingComplete) {  // Only show error if not completed
                    this.logMessage('Connection error occurred', 'error');
                }
                this.eventSource.close();
            };
            // this.eventSource.onerror = (error) => {
            //     // Only log error if processing didn't complete successfully
            //     if (!this.processingComplete) {
            //         console.error('SSE error:', error);
            //         this.logMessage('Connection error occurred', 'error');
            //         this.showError('Connection to server lost. Please try again.');
            //     }
            //     if (this.eventSource) {
            //         this.eventSource.close();
            //     }
            // };

        } catch (error) {
            console.error('Pipeline error:', error);
            this.showError('Failed to start processing pipeline');
        }
    }

    handleProgressEvent(event) {
        try {
            const data = JSON.parse(event.data);
            this.updateProgress(data);

            if (data.type === 'complete') {
                this.processingComplete = true;
                // Close EventSource immediately to prevent error event
                if (this.eventSource) {
                    this.eventSource.close();
                    this.eventSource = null;
                }
                this.handleProcessingComplete(data);
            } else if (data.type === 'error') {
                this.handleProcessingError(data);
            }
        } catch (error) {
            console.error('Error parsing progress event:', error);
        }
    }

    updateProgress(data) {
        const progressFill = document.getElementById('progress-fill');
        const progressPercent = document.getElementById('progress-percent');
        const progressMessage = document.getElementById('progress-message');

        if (data.progress !== undefined) {
            progressFill.style.width = `${data.progress}%`;
            progressPercent.textContent = `${data.progress}%`;
        }

        if (data.message) {
            progressMessage.textContent = data.message;
            this.logMessage(data.message, 'info');
        }
    }

    handleProcessingComplete(data) {
        this.processingComplete = true;  // Set flag BEFORE closing
        this.showLoading(false);
        if (this.eventSource) {
            this.eventSource.close();
        }

        this.logMessage('Processing completed successfully!', 'success');
        this.switchSection('results-section');

        // Store output file names for downloads
        this.outputFile = data.output_file;
        this.reportFile = data.report_file;
    }

    handleProcessingError(data) {
        this.showLoading(false);
        this.eventSource.close();
        this.showError(data.message);
    }

    initializeProgress() {
        document.getElementById('progress-fill').style.width = '0%';
        document.getElementById('progress-percent').textContent = '0%';
        document.getElementById('progress-message').textContent = 'Initializing...';
        document.getElementById('log-container').innerHTML = '';
    }

    logMessage(message, type = 'info') {
        const logContainer = document.getElementById('log-container');
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;

        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    async downloadFile(type) {
        try {
            const filename = type === 'dataset' ? this.outputFile : this.reportFile;
            if (!filename) {
                this.showError('File not available for download');
                return;
            }

            const endpoint = type === 'dataset' ? 'dataset' : 'report';
            const response = await fetch(`${this.apiBaseUrl}/api/download/${endpoint}/${encodeURIComponent(filename)}`);

            if (!response.ok) {
                throw new Error('Download failed');
            }

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            this.logMessage(`${type} downloaded successfully`, 'success');

        } catch (error) {
            console.error('Download error:', error);
            this.showError('Download failed');
        }
    }

    switchSection(sectionId) {
        // Hide all sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });

        // Show target section
        document.getElementById(sectionId).classList.add('active');
    }

    showError(message) {
        document.getElementById('error-message').textContent = message;
        this.switchSection('error-section');
        this.showLoading(false);
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        if (show) {
            overlay.classList.remove('hidden');
        } else {
            overlay.classList.add('hidden');
        }
    }

    resetApp() {
        this.currentFile = null;
        this.outputFile = null;
        this.reportFile = null;

        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }

        this.removeFile();
        this.switchSection('upload-section');
        this.showLoading(false);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DataCleanerApp();
});
