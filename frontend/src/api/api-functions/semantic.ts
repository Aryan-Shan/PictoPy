import { apiClient } from '../axiosConfig';

interface BackendRes<T = any> {
    success: boolean;
    error?: string;
    message?: string;
    data?: T;
}

export const semanticSearch = async (query: string): Promise<BackendRes<any>> => {
    const response = await apiClient.get(`/semantic/search?q=${encodeURIComponent(query)}`);
    return response.data;
};

export const triggerIndexing = async (): Promise<BackendRes<{ message: string }>> => {
    const response = await apiClient.post('/semantic/index');
    return response.data;
};

export interface IndexingStatus {
    is_active: boolean;
    current: number;
    total: number;
    error: string | null;
}

export const getIndexingStatus = async (): Promise<BackendRes<IndexingStatus>> => {
    const response = await apiClient.get('/semantic/status');
    return response.data;
};
