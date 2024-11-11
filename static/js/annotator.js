class AnnotatorUI {
    constructor() {
        this.evtSource = new EventSource('/stream');
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Process button
        $('#processBtn').click(() => this.handleProcess());
        
        // Refresh button
        $('#refreshBtn').click(() => this.handleRefresh());
        
        // SSE events
        this.evtSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateUI(data);
        };
    }

    updateUI(data) {
        if (data.is_processing) {
            // Show progress
            $('#progress').show();
            const progress = (data.processed / data.total) * 100;
            $('.progress-bar').css('width', progress + '%');
            $('#progress-text').text(`Processing: ${data.current_image} (${data.processed}/${data.total})`);
            
            // Update stats
            this.updateStats();
            
            // Refresh visualizations
            this.refreshVisualizations();
        } else {
            $('#progress').hide();
            $('#progress-text').text('');
        }
    }

    updateStats() {
        $.get('/status', (data) => {
            // Update total counts
            $('#total-images').text(
                data.train.total_images + 
                data.val.total_images + 
                data.test.total_images
            );
            $('#total-processed').text(
                data.train.processed_images + 
                data.val.processed_images + 
                data.test.processed_images
            );
            $('#total-viz').text(
                data.train.visualizations + 
                data.val.visualizations + 
                data.test.visualizations
            );
        });
    }

    refreshVisualizations() {
        const timestamp = new Date().getTime();
        $('.visualization').each(function() {
            let src = $(this).attr('src');
            if (src) {
                src = src.split('?')[0];
                $(this).attr('src', src + '?t=' + timestamp);
                $(this).parent('a').attr('href', src + '?t=' + timestamp);
            }
        });
    }

    handleProcess() {
        const btn = $('#processBtn');
        btn.prop('disabled', true);
        $('#progress').show();
        
        $.post('/process', (data) => {
            if (data.status === 'success') {
                this.refreshVisualizations();
                setTimeout(() => {
                    location.reload();
                }, 1000);
            } else {
                alert('Error: ' + data.error);
            }
        })
        .fail(() => {
            alert('Processing failed!');
        })
        .always(() => {
            btn.prop('disabled', false);
            $('#progress').hide();
        });
    }

    handleRefresh() {
        const btn = $('#refreshBtn');
        btn.html('<i class="fas fa-sync-alt fa-spin me-2"></i>Refreshing...');
        
        this.refreshVisualizations();
        this.updateStats();
        
        setTimeout(() => {
            btn.html('<i class="fas fa-sync-alt me-2"></i>Refresh');
        }, 1000);
    }
}

// Initialize when document is ready
$(document).ready(() => {
    // Initialize Lightbox
    lightbox.option({
        'resizeDuration': 200,
        'wrapAround': true,
        'albumLabel': 'Image %1 of %2'
    });

    // Initialize UI
    const ui = new AnnotatorUI();
}); 