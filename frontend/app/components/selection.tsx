"use client";
import { useBrawler } from './brawler-context';
import React, { useMemo, useEffect, useState } from "react";
import {Dropdown, DropdownTrigger, DropdownMenu, DropdownItem, Button} from "@nextui-org/react";
import type { Selection } from "@nextui-org/react";
import { ChevronDown, Check, Info, X } from "lucide-react";
import { motion, AnimatePresence } from 'framer-motion';
import Image from "next/image";

function Menu() {
  const { selectedMap, availableMaps, maps, availableGameModes, mapSelectionSetup } = useBrawler();
  const [selectedKeys, setSelectedKeys] = React.useState<Selection>(new Set(selectedMap ? [selectedMap] : []));
  const [selectedGameMode, setSelectedGameMode] = React.useState<string>("");

  const handleSelectionChange = (keys: Selection) => {
    setSelectedKeys(keys);
    const selected = Array.from(keys)[0] as string;
    mapSelectionSetup(selected);
  };
  if (!maps) {
    return <div>Loading...</div>;
  };
  console.log("maps: ", maps);

  const filteredMaps = useMemo(() => {
    if (selectedGameMode === "") {
      return availableMaps.sort();
    } else {
      return availableMaps.filter(map => maps.maps[map]?.game_mode === selectedGameMode).sort();
    }
  }, [availableMaps, maps, selectedGameMode]);

  return (
    <div className="w-64 relative">
      <div className="absolute -top-3 left-4 z-20">
        <span className="px-2 py-0.5 bg-yellow-300 text-gray-900 text-sm rounded-xl">Map</span>
      </div>
      <Dropdown>
        <DropdownTrigger>
          <Button 
            variant="flat" 
            className="w-full justify-between bg-gray-700 text-white border border-gray-900 rounded-2xl h-[55px] items-center flex"
          >
            <span className="capitalize">{selectedMap || "Select Map"}</span>
            <ChevronDown className="text-gray-400" size={20} />
          </Button>
        </DropdownTrigger>
        <DropdownMenu 
          aria-label="Map selection"
          variant="flat"
          disallowEmptySelection
          selectionMode="single"
          selectedKeys={selectedKeys}
          onSelectionChange={handleSelectionChange}
          className="bg-gray-700 text-white p-2 rounded-xl max-h-64 overflow-auto custom-scrollbar"
        >
          {filteredMaps.map((map) => (
            <DropdownItem key={map} className="text-white hover:bg-gray-600 px-2 py-2 rounded-xl gap-x-2">
              <div className="flex items-center gap-2">
                {maps.maps[map]?.game_mode ? (
                  <Image
                    src={`/game_modes/${maps.maps[map].game_mode}.png`}
                    alt={map}
                    width={30}
                    height={30}
                    onError={(e) => {
                      e.currentTarget.src = '/brawlstar.png'; 
                    }}
                  />
                ) : (
                  <div className="w-[30px] h-[30px] bg-gray-600 rounded-full"></div>
                )}
                <span>{map}</span>
              </div>
            </DropdownItem>
          ))}
        </DropdownMenu>
      </Dropdown>
      <div className='flex items-center gap-1 mt-4'>
        {availableGameModes && availableGameModes.length > 0 ? (
          availableGameModes.map((game_mode) => (
            <motion.button
            key={game_mode}
            className={`bg-gray-200 p-1 min-w-[40px] h-[40px] rounded-xl ${selectedGameMode === game_mode ? 'ring-2 border-blue-500 border-2 ring-blue-500' : ''}`}
            whileHover={{ scale: 1.1, zIndex: 10, backgroundColor: "#9ca3af" }}
            whileTap={{ scale: 0.9, zIndex: 10, transition: { duration: 0.3 } }}
            onClick={() => setSelectedGameMode(prevMode => prevMode === game_mode ? "" : game_mode)}
          >
            <div className="relative w-full h-full">
              <Image
                src={`/game_modes/${game_mode}.png`}
                alt={game_mode}
                layout="fill"
                objectFit="contain"
                onError={(e) => {
                  e.currentTarget.src = '/brawlstar.png'; 
                }}
              />
            </div>
          </motion.button>
          ))
        ) : (
          <div> Loading... </div>
        )}
      </div>
    </div>
  );
}

const FilterByPlayer = () => {
  const { playerTagError, setPlayerTagError, setCurrentPlayer, setFilterPlayerBrawlers, setMinBrawlerLevel } = useBrawler();
  const [playerTag, setPlayerTag] = useState<string>("");
  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setPlayerTag(event.target.value);
  };

  const handleSubmit = () => {
    setPlayerTagError(false)
    setCurrentPlayer(playerTag);
  };

  useEffect(() => {
    if (playerTagError) {
      const timer = setTimeout(() => {
        setPlayerTagError(false);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [playerTagError]);

  return (
    <div className='flex flex-row'>
    <motion.button
          className="h-[45px] w-[45px] mt-[5px] mr-[10px] bg-gray-200 rounded-xl flex items-center justify-center"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Info size={24} />
        </motion.button>
    <div className='flex-col'>
      <div className={`p-2 rounded-2xl border border-gray-300 bg-gray-200 items-center justify-between flex  h-[55px]`}>
        <input
          type="text"
          placeholder="Player tag..."
          value={playerTag}
          onChange={handleInputChange}
          className={`p-2 rounded-xl mr-2 border ${playerTagError ? 'border-red-300 bg-red-200' : 'border-gray-300 bg-gray-100'} bg-gray-100 flex-grow`}
        />
        <motion.button 
          className={`rounded-xl p-2  ${playerTagError ? 'bg-red-300' : 'bg-green-300'}`}
          onClick={handleSubmit}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9, transition: { duration: 0.3 } }}
        >
          {playerTagError ?  <X/> : <Check />}
        </motion.button>
      </div>
      <PlayerTagFilters/>
    </div>
    </div>
  );
};

const ToggleSwitch = ( {isOn, toggleSwitch} ) => {
  const spring = {
    type: "spring",
    stiffness: 700,
    damping: 30
  };

  return (
    <div 
      className={`w-16 h-8 flex items-center rounded-full p-1 cursor-pointer ${
        isOn ? 'bg-green-400' : 'bg-red-400'
      }`} 
      onClick={toggleSwitch}
    >
      <motion.div 
        className={`w-6 h-6 rounded-full ${
          isOn ? 'bg-white' : 'bg-gray-200'
        }`} 
        layout
        transition={spring}
        animate={{ x: isOn ? 32 : 0 }}
      />
    </div>
  );
};


const PlayerTagFilters = () => {
  const {filterPlayerBrawlers, minBrawlerLevel, setMinBrawlerLevel, setFilterPlayerBrawlers} = useBrawler();
  const levels = [9,10,11]
  if (filterPlayerBrawlers === null) {
    return
  }

  return (
    <div className='mt-5 flex flex-row items-center justify-start relative'>
      <div className="flex flex-col items-center justify-center mr-4 w-20">
        <ToggleSwitch 
          isOn={filterPlayerBrawlers} 
          toggleSwitch={() => setFilterPlayerBrawlers(!filterPlayerBrawlers)} 
        />
        <span className="font-bold text-center mt-1">
          {filterPlayerBrawlers ? "Filter On" : "Filter Off"}
        </span>
      </div>
      <div className="flex-1">
        <div className="relative">
          <div className="absolute -top-4 left-1 z-20">
            <span className="px-2 py-0.5 bg-green-300 text-gray-900 text-sm rounded-xl">Min Brawler Level</span>
          </div>
          <div className='flex items-center justify-center gap-4 font-bold bg-gray-300 rounded-2xl pt-3 pb-2 pr-2 pl-2'>
            {levels.map(LevelButton)}
          </div>
        </div>
      </div>
    </div>
  );
}

const LevelButton = (brawlerLevel: number) => {
  const {minBrawlerLevel, setMinBrawlerLevel} = useBrawler();
  return (
    <motion.button
    key={brawlerLevel}
    className={`bg-gray-200 p-1 min-w-[40px] h-[40px] rounded-xl ${minBrawlerLevel === brawlerLevel ? 'ring-2 border-blue-500 border-2 ring-blue-500' : ''}`}
    whileTap={{scale: 0.9, transition: { duration: 0.3 }}}
    whileHover={{ scale: 1.1, zIndex: 10, backgroundColor: "#9ca3af" }}
    onClick={() => setMinBrawlerLevel(brawlerLevel)}
    >
      {brawlerLevel}
    </motion.button>
  )
}


const Selection = () => {
  const { firstPick, setFirstPick, resetEverything } = useBrawler();

  return (
    <div className='w-full flex flex-col md:flex-row gap-x-12 py-3 gap-y-3 justify-center mb-16 md:mb-0'>
      <FilterByPlayer/>
      <Menu />
      <motion.button
        className='flex w-[150px] h-[55px] rounded-2xl justify-center items-center p-2 font-bold text-xl border-2'
        onClick={() => setFirstPick(!firstPick)}
        style={{backgroundColor: firstPick ? '#76a8f9' : '#f47c7c', borderColor: firstPick ? '#3b82f6' : '#ef4444'}}
        whileHover={{ scale: 1.1, zIndex: 10 }}
        whileTap={{ scale: 0.9, zIndex: 10, transition: { duration: 0.3 } }}
      >
        {firstPick ? 'First Pick' : 'Second Pick'}
      </motion.button>
      <motion.button
        className='flex w-[150px] h-[55px] rounded-2xl justify-center items-center p-2 font-bold text-xl border-2 bg-gray-200 border-gray-300'
        onClick={() => resetEverything()}
        whileHover={{ scale: 1.1, zIndex: 10 }}
        whileTap={{ scale: 0.9, zIndex: 10, transition: { duration: 0.3 } }}
      >
        Reset
      </motion.button>
    </div>
  );
};

export default Selection;