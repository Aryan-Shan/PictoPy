
import { apiClient } from '../axiosConfig';
import { searchEndpoints } from '../apiEndpoints';

export const searchImages = async (query: string, limit: number = 50) => {
    const response = await apiClient.get(searchEndpoints.search, {
        params: { q: query, limit },
    });
    return response.data;
};
