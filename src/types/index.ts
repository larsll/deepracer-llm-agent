export interface Image {
    id: string;
    url: string;
    format: string;
    size: number;
}

export interface ProcessedImage {
    id: string;
    originalUrl: string;
    processedUrl: string;
    status: string;
    error?: string;
}