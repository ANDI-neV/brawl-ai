import React, { createContext, useState, useContext, ReactNode, useEffect, useMemo, useCallback, useRef } from 'react';
import brawlerJson from "../../../backend/src/out/brawlers/brawlers.json";
import { fetchMaps, fetchBrawlers, predictBrawlers, getPickrate, MapInterface, Mapping, getMapping, getPlayerBrawlers } from './api-handler';
import axios from 'axios';


interface BrawlerPickerProps {
  name: string;
}

interface MapData {
  game_mode: string;
  img_url: string;
}

interface BrawlerContextType {
  selectedBrawlers: (BrawlerPickerProps | null)[];
  availableBrawlers: BrawlerPickerProps[];
  availableMaps: string[];
  selectedMap: string;
  firstPick: boolean;
  isPredicting: boolean;
  error: string | null;
  brawlerScores: { [key: string]: number };
  brawlerPickrates: { [key: string]: number};
  maps: MapInterface | null;
  brawlerMapping: Mapping;
  loadingMapping: boolean;
  selectedMapData: MapData;
  availableGameModes: string[];
  currentPlayer: string;
  currentPlayerBrawlers: string[];
  filterPlayerBrawlers: boolean | null;
  minBrawlerLevel: number;
  playerTagError: boolean;
  setFirstPick: (firstPick: boolean) => void;
  setSelectedMap: (map: string) => void;
  selectBrawler: (brawler: BrawlerPickerProps, slot: number) => void;
  clearSlot: (slot: number) => void;
  updatePredictions: (map: string, brawlers: string[], firstPick: boolean) => void;
  retrieveBrawlerPickrates: (map: string) => void;
  mapSelectionSetup: (map: string) => void;
  resetEverything: () => void;
  setCurrentPlayer: (player: string) => void;
  setCurrentPlayerBrawlers: (brawlers: string[]) => void;
  setMinBrawlerLevel: (brawlerLevel: number) => void;
  setFilterPlayerBrawlers: (filterPlayerBrawlers: boolean) => void;
  setPlayerTagError: (playerTagError: boolean) => void;
}

const BrawlerContext = createContext<BrawlerContextType | undefined>(undefined);

function get_brawlers(): BrawlerPickerProps[] {
  return Object.keys(brawlerJson).map(name => ({ name }));
}

export function BrawlerProvider({ children }: { children: ReactNode }) {
  const [selectedBrawlers, setSelectedBrawlers] = useState<(BrawlerPickerProps | null)[]>(Array(6).fill(null));
  const [firstPick, setFirstPick] = useState(true);
  const allBrawlers = useMemo(() => get_brawlers(), []);
  const [availableMaps, setAvailableMaps] = useState<string[]>([]);
  const [selectedMap, setSelectedMap] = useState<string>('');
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [brawlerScores, setBrawlerScores] = useState<{ [key: string]: number }>({});
  const [brawlerPickrates, setBrawlerPickrates] = useState<{ [key: string]: number}>({});
  const [maps, setMaps] = useState<MapInterface>({ maps: {} });
  const [selectedMapData, setSelectedMapData] = useState<MapData>({game_mode: '', img_url: ''});
  const [brawlerMapping, setBrawlerMapping] = useState<Mapping>({});
  const [loadingMapping, setLoadingMapping] = useState(true);
  const [availableGameModes, setAvailableGameModes] = useState<string[]>([]);
  const [currentPlayer, setCurrentPlayer] = useState<string>(() => {
    const savedPlayerTag = localStorage.getItem('playerTag');
    return savedPlayerTag || "";
  });
  const [minBrawlerLevel, setMinBrawlerLevel] = useState<number>(11);
  const [currentPlayerBrawlers, setCurrentPlayerBrawlers] = useState<string[]>([]);
  const [playerTagError, setPlayerTagError] = useState<boolean>(false);
  const [filterPlayerBrawlers, setFilterPlayerBrawlers] = useState<boolean | null>(null);

  useEffect(() => {
    const getMaps = async () => {
    try {
      const fetchedMaps = await fetchMaps();
      setMaps(fetchedMaps);
      setAvailableMaps(Object.keys(fetchedMaps.maps));
      console.log("maps available: ", Object.keys(fetchedMaps.maps))
      getGameModes(fetchedMaps);
    }
    catch( err ) {
      console.error("Error fetching maps:", err);
      setError("Failed to fetch maps");
    }
    };
    getMaps();
  }, []);

  useEffect(() => {
    if (currentPlayer) {
      localStorage.setItem('playerTag', currentPlayer);
    } else {
      localStorage.removeItem('playerTag');
    }
  }, [currentPlayer]);

  useEffect(() => {
    const getCurrentPlayerBrawlers = async () => {
      if (currentPlayer !== "") {
        console.log("Get current player brawlers for: ", currentPlayer);
        try {
          const player = currentPlayer.charAt(0) !== "#" ? "#" + currentPlayer : currentPlayer;
          const filteredBrawlers = await getPlayerBrawlers(player, minBrawlerLevel);
          setPlayerTagError(false)
          setCurrentPlayerBrawlers(filteredBrawlers.brawlers);
          setFilterPlayerBrawlers(true);
        } catch (error) {
          if (axios.isAxiosError(error) && error.response?.status === 404) {
            setPlayerTagError(true);
            setCurrentPlayer('');
            if (filterPlayerBrawlers !== null) {
              setFilterPlayerBrawlers(false)
            }
          } else {
            console.error("Error fetching filtered Brawlers:", error);
          }
        }
      }
    };
  
    getCurrentPlayerBrawlers();
  }, [currentPlayer, minBrawlerLevel]);

  const getGameModes = useCallback((maps: MapInterface) => {
    const filteredGameModes: string[] = []
    Object.values(maps.maps).map((map) => {
      if (!filteredGameModes.includes(map.game_mode)) {
        filteredGameModes.push(map.game_mode)
      }
    })
    console.log("Available game modes: ", filteredGameModes)
    setAvailableGameModes(filteredGameModes)
  }, []) 

  useEffect(() => {
    const fetchMapping = async () => {
      try {
        const mapping = await getMapping();
        setBrawlerMapping(mapping);
        setLoadingMapping(false);
      } catch (error) {
        setError('Failed to fetch brawler mapping');
        setLoadingMapping(false);
      }
    };

    fetchMapping();
  }, []);
    
  const pickratesFetchedRef = useRef(false);

  const selectedBrawlerNames = useMemo(
    () => selectedBrawlers.filter(Boolean).map(b => b!.name),
    [selectedBrawlers]
  );

  const resetEverything = useCallback(() => {
    setFirstPick(true)
    setSelectedBrawlers(Array(6).fill(null))
    setSelectedMap('')
    setIsPredicting(false)
    setError(null)
    pickratesFetchedRef.current = false
    setBrawlerScores({})
    setBrawlerPickrates({})
  }, [])

  const retrieveBrawlerPickrates = useCallback((map: string) => {
    if (pickratesFetchedRef.current) {
      console.info("Pickrates already fetched, skipping retrieval");
      return;
    }

    console.info("Retrieving pickrate for map:", map);
    if (map) {
      setError(null);
      getPickrate(map)
        .then(probabilities => {
          console.log("Received probabilities:", probabilities);
          if (probabilities && Object.keys(probabilities).length > 0) {
            setBrawlerPickrates(probabilities);
            pickratesFetchedRef.current = true;
          } else {
            console.warn("Received empty probabilities");
            setError("Received empty data from server");
          }
        })
        .catch(error => {
          console.error("Error retrieving pickrates:", error);
          setError("Failed to retrieve pickrates: " + error.message);
        });
    } else {
      console.warn("No map selected for pickrate retrieval");
    }
  }, []);
  
  const updatePredictions = useCallback((map: string, brawlers: string[], firstPick: boolean) => {
    console.log("Updating predictions:", { map, brawlers, firstPick });
    if (map) {
      setIsPredicting(true);
      setError(null);
      predictBrawlers(map, brawlers, firstPick)
        .then(probabilities => {
          console.log("Received probabilities: ", probabilities);
          setBrawlerScores(probabilities);
          setIsPredicting(false);
        })
        .catch(error => {
          console.error("Error predicting brawlers: ", error);
          setError("Failed to predict brawlers");
          setIsPredicting(false);
        });
    } else {
      console.warn("No map selected for prediction.");
    }
  }, []);


  const mapSelectionSetup = useCallback((map: string) => {
    console.log('Map selected:', map);
    setSelectedMap(map);

    setSelectedMapData(maps?.maps[map]);
    if (!pickratesFetchedRef.current) {
      retrieveBrawlerPickrates(map);
    }
    updatePredictions(map, selectedBrawlers.filter(Boolean).map(b => b!.name), firstPick);
  }, [selectedBrawlers, firstPick, updatePredictions]);

  const selectBrawler = useCallback((brawler: BrawlerPickerProps, slot: number) => {
    setSelectedBrawlers(prev => {
      const newSelection = [...prev];
      newSelection[slot] = brawler;
      return newSelection;
    });
  }, []);

  const clearSlot = useCallback((slot: number) => {
    setSelectedBrawlers(prev => {
      const newSelection = [...prev];
      newSelection[slot] = null;
      return newSelection;
    });
  }, []);

  useEffect(() => {
    if (selectedMap) {
      updatePredictions(selectedMap, selectedBrawlers.filter(Boolean).map(b => b!.name), firstPick);
    }
  }, [selectedBrawlers, selectedMap, firstPick, updatePredictions]);


  const availableBrawlers = useMemo(() => {
    const selectedBrawlerNamesSet = new Set(selectedBrawlerNames);
    return allBrawlers.filter(brawler => !selectedBrawlerNamesSet.has(brawler.name));
  }, [selectedBrawlerNames, allBrawlers]);

  return (
    <BrawlerContext.Provider value={{
      selectedBrawlers,
      availableBrawlers,
      availableMaps,
      selectedMap,
      firstPick,
      isPredicting,
      error,
      brawlerScores,
      brawlerPickrates,
      maps,
      brawlerMapping,
      loadingMapping,
      selectedMapData,
      availableGameModes,
      currentPlayer,
      currentPlayerBrawlers,
      filterPlayerBrawlers,
      minBrawlerLevel,
      playerTagError,
      setFirstPick,
      setSelectedMap,
      selectBrawler,
      clearSlot,
      updatePredictions,
      retrieveBrawlerPickrates,
      mapSelectionSetup,
      resetEverything,
      setCurrentPlayer,
      setCurrentPlayerBrawlers,
      setMinBrawlerLevel,
      setFilterPlayerBrawlers,
      setPlayerTagError
    }}>
      {children}
    </BrawlerContext.Provider>
  );
}

export function useBrawler() {
  const context = useContext(BrawlerContext);
  if (context === undefined) {
    throw new Error('useBrawler must be used within a BrawlerProvider');
  }
  return context;
}
