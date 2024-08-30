import axios from 'axios';

const API_URL = 'http://localhost:7000';

export const fetchMaps = async () => {
  const response = await axios.get(`${API_URL}/maps`);
  return response.data.maps;
};

export const fetchBrawlers = async () => {
  const response = await axios.get(`${API_URL}/brawlers`);
  return response.data.brawlers as string[];
};

export const predictBrawlers = async (map: string, brawlers: string[]) => {
  const response = await axios.post(`${API_URL}/predict`, { map, brawlers });
  return response.data.probabilities;
};