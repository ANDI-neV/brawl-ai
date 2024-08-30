import React, { createContext, useState, useContext, ReactNode, useEffect, useMemo } from 'react';
import brawlerJson from "../../../backend/src/out/brawlers/brawlers.json";
import { fetchMaps } from './api-handler';

interface BrawlerPickerProps {
  name: string;
  score: number;
  pickrate: number;
}

interface BrawlerContextType {
  selectedBrawlers: (BrawlerPickerProps | null)[];
  availableBrawlers: BrawlerPickerProps[];
  availableMaps: string[];
  selectedMap: string;
  firstPick: boolean;
  setFirstPick: (firstPick: boolean) => void;
  setSelectedMap: (map: string) => void;
  selectBrawler: (brawler: BrawlerPickerProps, slot: number) => void;
  clearSlot: (slot: number) => void;
}

const BrawlerContext = createContext<BrawlerContextType | undefined>(undefined);

function get_brawlers(): BrawlerPickerProps[] {
  return Object.keys(brawlerJson).map(name => ({
    name,
    score: -1,
    pickrate: -1
  }));
}

export function BrawlerProvider({ children }: { children: ReactNode }) {
  const [selectedBrawlers, setSelectedBrawlers] = useState<(BrawlerPickerProps | null)[]>(Array(6).fill(null));
  const [firstPick, setFirstPick] = useState(true);
  const allBrawlers = useMemo(() => get_brawlers(), []);
  const [availableMaps, setAvailableMaps] = useState<string[]>([]);
  const [selectedMap, setSelectedMap] = useState<string>('');

  useEffect(() => {
    fetchMaps().then(setAvailableMaps);
  }, []);

  const availableBrawlers = useMemo(() => {
    const selectedBrawlerNames = selectedBrawlers.filter(Boolean).map(b => b!.name);
    return allBrawlers.filter(brawler => !selectedBrawlerNames.includes(brawler.name));
  }, [selectedBrawlers, allBrawlers]);

  const selectBrawler = (brawler: BrawlerPickerProps, slot: number) => {
    setSelectedBrawlers(prev => {
      const newSelection = [...prev];
      newSelection[slot] = brawler;
      return newSelection;
    });
  };

  const clearSlot = (slot: number) => {
    setSelectedBrawlers(prev => {
      const newSelection = [...prev];
      newSelection[slot] = null;
      return newSelection;
    });
  };

  return (
    <BrawlerContext.Provider value={{ 
      selectedBrawlers, 
      availableBrawlers, 
      availableMaps,
      selectedMap,
      firstPick, 
      setFirstPick, 
      setSelectedMap,
      selectBrawler, 
      clearSlot 
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