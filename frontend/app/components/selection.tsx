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
  const [selectedGameMode, setSelectedGameMode] = React.useState<string>("");

  const filteredMaps = useMemo(() => (
    selectedGameMode === ""
      ? availableMaps.sort()
      : availableMaps.filter(map => maps?.maps[map]?.game_mode === selectedGameMode).sort()
  ), [availableMaps, maps, selectedGameMode]);

  const handleSelectionChange = (keys: Selection) => {
    const selected = Array.from(keys)[0] as string;
    mapSelectionSetup(selected);
  };

  if (!maps) {
    return <div>Loading...</div>;
  }

  console.log("maps: ", maps);

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
          selectedKeys={selectedMap ? new Set([selectedMap]) : new Set()}
          onSelectionChange={handleSelectionChange}
          className="bg-gray-700 text-white p-2 rounded-xl max-h-64 overflow-auto custom-scrollbar"
        >
          {filteredMaps.map((map) => (
            <DropdownItem key={map} textValue={map} className="text-white hover:bg-gray-600 px-2 py-2 rounded-xl gap-x-2">
              <div className="flex items-center gap-2">
                {maps.maps[map]?.game_mode ? (
                  <Image
                    src={`/game_modes/${maps.maps[map].game_mode}.webp`}
                    alt={map}
                    width={30}
                    height={30}
                    style={{
                      width: 'auto',
                      height: 'auto',
                      objectFit: 'contain'
                    }}
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
                src={`/game_modes/${game_mode}.webp`}
                alt={game_mode}
                fill={true}
                style={{objectFit: "contain"}}
                sizes="(max-width: 768px) 100vw, 33vw"
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

const PlayerTagInformation = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="relative inline-block">
      <motion.nav
        initial={false}
        animate={isOpen ? "open" : "closed"}
        className="menu"
      >
        <motion.button
          className="h-[45px] w-[45px] mt-[5px] mr-[10px] bg-gray-200 rounded-xl flex items-center justify-center relative z-10"
          whileHover={{ scale: 1.05 }}
          onClick={() => setIsOpen(!isOpen)}
          whileTap={{ scale: 0.95 }}
        >
          <Info size={24} />
        </motion.button>
        <motion.div
          className="absolute bottom-full transform -translate-x-1/2 mb-2 z-20"
          variants={{
            open: {
              opacity: 1,
              y: 0,
              display: "block",
            },
            closed: {
              opacity: 0,
              y: 10,
              transitionEnd: {
                display: "none",
              },
            },
          }}
          transition={{ duration: 0.2 }}
        >
          <div className="w-64 rounded-xl bg-gray-300 p-2 shadow-lg text-sm">
            Enter your Player Tag to receive only suggestions based on the Brawlers you have. You may also filter by the Minimum Level of Brawlers you want displayed
          </div>
        </motion.div>
      </motion.nav>
    </div>
  );
}



const FilterByPlayer = () => {
  const { 
    playerTagError, 
    setPlayerTagError, 
    setCurrentPlayer, 
    currentPlayer, 
    setFilterPlayerBrawlers, 
    setMinBrawlerLevel 
  } = useBrawler();
  const [playerTag, setPlayerTag] = useState(currentPlayer);
  
  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setPlayerTag(event.target.value);
  };

  const handleSubmit = () => {
    setPlayerTagError(false);
    setCurrentPlayer(playerTag);
  };

  useEffect(() => {
    if (playerTagError) {
      const timer = setTimeout(() => {
        setPlayerTagError(false);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [playerTagError, setPlayerTagError]);

  return (
    <div className="flex items-start">
      <PlayerTagInformation />
      <div className="flex-grow">
        <div className={`p-2 rounded-2xl border border-gray-300 bg-gray-200 flex items-center h-[55px]`}>
          <input
            type="text"
            placeholder="Player tag..."
            value={playerTag}
            onChange={handleInputChange}
            className={`p-2 rounded-xl mr-2 border ${playerTagError ? 'border-red-300 bg-red-200' : 'border-gray-300 bg-gray-100'} flex-grow`}
          />
          <motion.button 
            className={`rounded-xl p-2 ${playerTagError ? 'bg-red-300' : 'bg-green-300'}`}
            onClick={handleSubmit}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9, transition: { duration: 0.3 } }}
          >
            {playerTagError ? <X /> : <Check />}
          </motion.button>
        </div>
        <PlayerTagFilters />
      </div>
    </div>
  );
};

type ToggleSwitchProps = {
  isOn: boolean;
  toggleSwitch: () => void;
};

const ToggleSwitch = ({ isOn, toggleSwitch }: ToggleSwitchProps) => {
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
      onClick={() => {
        console.log("Toggling switch. Current value:", isOn);
        toggleSwitch();
      }}
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
  const { 
    filterPlayerBrawlers, 
    setFilterPlayerBrawlers,
    minBrawlerLevel,
    setMinBrawlerLevel 
  } = useBrawler();
  
  const levels = [9, 10, 11];

  const renderLevelButton = (brawlerLevel: number) => (
    <motion.button
      key={brawlerLevel}
      className={`bg-gray-200 p-1 min-w-[40px] h-[40px] rounded-xl ${
        minBrawlerLevel === brawlerLevel ? 'ring-2 border-blue-500 border-2 ring-blue-500' : ''
      }`}
      whileTap={{scale: 0.9, transition: { duration: 0.3 }}}
      whileHover={{ scale: 1.1, zIndex: 10, backgroundColor: "#9ca3af" }}
      onClick={() => setMinBrawlerLevel(brawlerLevel)}
    >
      {brawlerLevel}
    </motion.button>
  );

  if (filterPlayerBrawlers === null) {
    return null;
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
            <span className="px-2 py-0.5 bg-green-300 text-gray-900 text-sm rounded-xl">
              Min Brawler Level
            </span>
          </div>
          <div className='flex items-center justify-center gap-4 font-bold bg-gray-300 rounded-2xl pt-3 pb-2 pr-2 pl-2'>
            {levels.map(renderLevelButton)}
          </div>
        </div>
      </div>
    </div>
  );
};



const Selection = () => {
  const { firstPick, setFirstPick, resetEverything } = useBrawler();

  return (
    <div className='w-full flex flex-col items-center lg:items-stretch lg:flex-row gap-x-12 py-3 gap-y-3 justify-center mb-16 lg:mb-0'>
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