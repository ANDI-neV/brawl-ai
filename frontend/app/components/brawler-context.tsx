import React, { createContext, useState, useContext, ReactNode, useEffect, useMemo, useCallback, useRef } from 'react';
import brawlerJson from "../../../backend/src/out/brawlers/brawlers.json";
import { fetchMaps, fetchBrawlers, predictBrawlers, getPickrate, MapInterface, Mapping, getMapping } from './api-handler';

interface BrawlerPickerProps {
  name: string;
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
  setFirstPick: (firstPick: boolean) => void;
  setSelectedMap: (map: string) => void;
  selectBrawler: (brawler: BrawlerPickerProps, slot: number) => void;
  clearSlot: (slot: number) => void;
  updatePredictions: (map: string, brawlers: string[], firstPick: boolean) => void;
  retrieveBrawlerPickrates: (map: string) => void;
  mapSelectionSetup: (map: string) => void;
  resetEverything: () => void;
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
  const [maps, setMaps] = useState<MapInterface | null>(null);
  const [brawlerMapping, setBrawlerMapping] = useState<Mapping>({});
  const [loadingMapping, setLoadingMapping] = useState(true);

  useEffect(() => {
    const getMaps = async () => {
    try {
      const fetchedMaps = await fetchMaps();
      setMaps(fetchedMaps);
      setAvailableMaps(Object.keys(fetchedMaps.maps));
    }
    catch( err ) {
      console.error("Error fetching maps:", err);
      setError("Failed to fetch maps");
    }
    };
    getMaps();
  }, []);

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
      setFirstPick,
      setSelectedMap,
      selectBrawler,
      clearSlot,
      updatePredictions,
      retrieveBrawlerPickrates,
      mapSelectionSetup,
      resetEverything
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
