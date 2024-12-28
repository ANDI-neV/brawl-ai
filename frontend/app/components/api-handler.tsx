import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:7001';

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export interface MapInterface {
  maps: { [key: string]: {
    game_mode: string;
    img_url: string;
  } };
}

export const fetchMaps = async (): Promise<MapInterface> => {
  try {
    const response = await axios.get(`${API_URL}/maps`);
    return response.data;
  } catch (error) {
    console.error("Error fetching maps:", error);
    throw error;
  }
};

export const fetchBrawlers = async (): Promise<string[]> => {
  const response = await axios.get(`${API_URL}/brawlers`);
  return response.data.brawlers;
};


interface PredictionResponse {
  probabilities: { [key: string]: number };
}

export const predictBrawlers = async (map: string, brawlers: string[], firstPick: boolean, retries = 3): Promise<{ [key: string]: number }> => {
  try {
    console.log("try prediction!")
    const response = await axios.post<PredictionResponse>(`${API_URL}/predict`, { map, brawlers, first_pick: firstPick });
    return response.data.probabilities;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.status === 429 && retries > 0) {
      console.log(`Rate limited, retrying in 0.5 seconds... (${retries} retries left)`);
      await delay(500);
      return predictBrawlers(map, brawlers, firstPick, retries - 1);
    }
    if (axios.isAxiosError(error) && error.response?.status === 500) {
      return {};
    }
    throw error;
  }
};

interface PickrateResponse {
  pickrate: {[key: string]: number}
}

export const getPickrate = async (map: string, retries = 3): Promise<{ [key: string]: number}> => {
  try {
    const response = await axios.post<PickrateResponse>(`${API_URL}/pickrate`, {map});
    return response.data.pickrate;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.status === 429 && retries > 0 ) {
      console.log(`Rate limited, retrying in 0.5 seconds... (${retries} retries left)`);
      await delay(500);
      return getPickrate(map, retries - 1);
    }
    throw error;
  }
}

export interface Mapping {
  [key: string]: number;
}


export const getMapping = async (retries = 3): Promise<Mapping> => {
  try {
    const response = await axios.get(`${API_URL}/brawler-mapping`);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.status === 429 && retries > 0 ) {
      console.log(`Rate limited, retrying in 0.5 seconds... (${retries} retries left)`);
      await delay(500);
      return getMapping(retries - 1);
    }
    throw error;
  }
}

interface FilteredPlayerBrawlers {
  brawlers: string[]
}

export const getPlayerBrawlers = async (playerTag: string, minLevel: number, retries = 3): Promise<FilteredPlayerBrawlers> => {
  try {
    const response = await axios.post<FilteredPlayerBrawlers>(`${API_URL}/filtered-player-brawlers`, {player_tag: playerTag, min_level: minLevel});
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error) && error.response?.status === 429 && retries > 0 ) {
      console.log(`Rate limited, retrying in 0.5 seconds... (${retries} retries left)`);
      await delay(500);
      return getPlayerBrawlers(playerTag, minLevel, retries - 1);
    }
    throw error;
  }
}