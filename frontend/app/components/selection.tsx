"use client";
import { useBrawler } from './brawler-context';
import React, { useMemo, useEffect, useState } from "react";
import {Dropdown, DropdownTrigger, DropdownMenu, DropdownItem, Button} from "@nextui-org/react";
import type { Selection } from "@nextui-org/react";
import { ChevronDown, Check } from "lucide-react";
import { motion } from "framer-motion";
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
            className="w-full justify-between bg-gray-800 text-white border border-gray-700 rounded-xl h-[50px] items-center flex"
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
          className="bg-gray-800 text-white p-2 rounded-xl max-h-64 overflow-auto custom-scrollbar"
        >
          {filteredMaps.map((map) => (
            <DropdownItem key={map} className="text-white hover:bg-gray-700 px-2 py-2 rounded-xl gap-x-2">
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
  const { currentPlayer, filterPlayerBrawlers, setCurrentPlayer, setFilterPlayerBrawlers, setMinBrawlerLevel } = useBrawler();
  const [playerTag, setPlayerTag] = useState<string>("");

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setPlayerTag(event.target.value);
  };

  const handleSubmit = () => {
    setCurrentPlayer(playerTag);
  };

  return (
    <div className='flex-col'>
      <div className='p-2 rounded-2xl border border-gray-300 items-center justify-between flex bg-gray-200'>
        <input
          type="text"
          placeholder="Player tag..."
          value={playerTag}
          onChange={handleInputChange}
          className='p-2 rounded-xl mr-2 bg-gray-100 flex-grow'
        />
        <motion.button 
          className='rounded-xl p-2 bg-green-300'
          onClick={handleSubmit}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9, transition: { duration: 0.3 } }}
        >
          <Check />
        </motion.button>
      </div>
      <div className='p-2'>
        <motion.button 
          className='rounded-xl p-2 bg-green-300 border-2'
          onClick={() => setFilterPlayerBrawlers(!filterPlayerBrawlers)}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9, transition: { duration: 0.3 } }}
          style={{backgroundColor: filterPlayerBrawlers ? '#86efac' : '#f7798e', borderColor: filterPlayerBrawlers ? '#4ade80' : '#fe098e'}}
        >
          {filterPlayerBrawlers ? "filter on" : "filter off"} 
        </motion.button>
      </div>
    </div>
  );
};


const Selection = () => {
  const { firstPick, setFirstPick, resetEverything } = useBrawler();

  return (
    <div className='w-full flex flex-col md:flex-row gap-x-12 py-3 items-center gap-y-3 justify-center mb-16 md:mb-0'>
      <FilterByPlayer/>
      <Menu />
      <motion.button
        className='flex w-[150px] h-[50px] rounded-xl justify-center items-center p-2 font-bold text-xl border-2'
        onClick={() => setFirstPick(!firstPick)}
        style={{backgroundColor: firstPick ? '#76a8f9' : '#f7798e', borderColor: firstPick ? '#3b82f6' : '#ef4444'}}
        whileHover={{ scale: 1.1, zIndex: 10 }}
        whileTap={{ scale: 0.9, zIndex: 10, transition: { duration: 0.3 } }}
      >
        {firstPick ? 'First Pick' : 'Second Pick'}
      </motion.button>
      <motion.button
        className='flex w-[150px] h-[50px] rounded-xl justify-center items-center p-2 font-bold text-xl border-2'
        onClick={() => resetEverything()}
        style={{backgroundColor: '#9e9191', borderColor: '#304091'}}
        whileHover={{ scale: 1.1, zIndex: 10 }}
        whileTap={{ scale: 0.9, zIndex: 10, transition: { duration: 0.3 } }}
      >
        Reset
      </motion.button>
    </div>
  );
};

export default Selection;