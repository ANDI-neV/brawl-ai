import React, { createContext, useState, useContext, ReactNode, useEffect, useMemo, useCallback, useRef } from 'react';
import brawlerJson from "../../../backend/src/out/brawlers/brawlers.json";
import { fetchMaps, fetchBrawlers, predictBrawlers, getPickrate } from './api-handler';

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
  setFirstPick: (firstPick: boolean) => void;
  setSelectedMap: (map: string) => void;
  selectBrawler: (brawler: BrawlerPickerProps, slot: number) => void;
  clearSlot: (slot: number) => void;
  updatePredictions: () => void;
  retrieveBrawlerPickrates: () => void;
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

  useEffect(() => {
    fetchMaps().then(setAvailableMaps).catch(err => {
      console.error("Error fetching maps:", err);
      setError("Failed to fetch maps");
    });
  }, []);

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
    //TODO: set scores and pichrates to null
  }, [])

  const mapSelectionSetup = useCallback((map: string) => {
    setSelectedMap(map)
    // wait here on change of selectedMap, else race conditions
    retrieveBrawlerPickrates();
    updatePredictions();
  },[]
)

  const retrieveBrawlerPickrates = useCallback(() => {
    console.info("try retrieving pickrate")
    if (selectedMap) {
      setError(null);
      getPickrate(selectedMap).then(probabilities => {
        setBrawlerPickrates(probabilities);
      })
      .catch(error => {
        console.error("Error predicting brawlers:", error);
        setError("Failed to predict brawlers");
      })
    } else {
      console.warn("no map was selected.")
    }
  }, []

  )

  const updatePredictions = useCallback(() => {
    console.info("try predicting")
    if (selectedMap) {
      setIsPredicting(true);
      setError(null);
      predictBrawlers(selectedMap, selectedBrawlerNames, firstPick)
        .then(probabilities => {
          setBrawlerScores(probabilities);
          setIsPredicting(false);
        })
        .catch(error => {
          console.error("Error predicting brawlers: ", error);
          setError("Failed to predict brawlers");
          setIsPredicting(false);
        });
    } else {
      console.warn("no map was selected.")
    }
  }, [selectedMap, selectedBrawlerNames, firstPick, isPredicting]);

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
