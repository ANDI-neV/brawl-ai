"use client";
import { useBrawler } from './brawler-context';
import React, { useMemo, useEffect, useState } from "react";
import { ChevronDown, Check, Info, X } from "lucide-react";
import { motion } from 'framer-motion';
import Image from "next/image";

const GAME_MODE_ICON_SRC: Record<string, string> = {
  "Bounty": "/game_modes/Bounty.webp",
  "Brawl Ball": "/game_modes/Brawl Ball.webp",
  "Brawl Hockey": "/game_modes/Brawl Hockey.webp",
  "Cleaning Duty": "/game_modes/Cleaning Duty.webp",
  "Gem Grab": "/game_modes/Gem Grab.webp",
  "Heist": "/game_modes/Heist.webp",
  "Hot Zone": "/game_modes/Hot Zone.webp",
  "Knockout": "/game_modes/Knockout.webp",
};

function getGameModeIconSrc(gameMode: string) {
  return GAME_MODE_ICON_SRC[gameMode] || "/brawlstar.png";
}

function Menu() {
  const { selectedMap, availableMaps, maps, availableGameModes, mapSelectionSetup, error } = useBrawler();
  const [selectedGameMode, setSelectedGameMode] = React.useState<string>("");

  const filteredMaps = useMemo(() => (
    selectedGameMode === ""
      ? [...availableMaps].sort()
      : availableMaps.filter(map => maps?.maps[map]?.game_mode === selectedGameMode).sort()
  ), [availableMaps, maps, selectedGameMode]);

  if (!maps) {
    return <div>Loading...</div>;
  }

  return (
    <div className="w-64 relative">
      <div className="absolute -top-3 left-4 z-20">
        <span className="px-2 py-0.5 bg-yellow-300 text-gray-900 text-sm rounded-xl">Map</span>
      </div>
      <div className="relative">
        <select
          aria-label="Map selection"
          disabled={availableMaps.length === 0}
          value={selectedMap}
          onChange={(event) => mapSelectionSetup(event.target.value)}
          className="w-full appearance-none rounded-2xl border border-gray-900 bg-gray-700 px-4 pr-12 text-white h-[55px] outline-none disabled:cursor-not-allowed disabled:opacity-60"
        >
          <option value="" disabled>
            {filteredMaps.length > 0 ? "Select Map" : "No maps available"}
          </option>
          {filteredMaps.map((map) => (
            <option key={map} value={map}>
              {map}
            </option>
          ))}
        </select>
        <ChevronDown
          className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-gray-300"
          size={20}
        />
      </div>
      <div className='flex items-center gap-1 mt-4'>
        {availableGameModes && availableGameModes.length > 0 ? (
          availableGameModes.map((game_mode) => (
            <motion.button
            key={game_mode}
            className={`bg-gray-200 p-1 w-[40px] h-[40px] rounded-xl flex items-center justify-center overflow-hidden ${selectedGameMode === game_mode ? 'ring-2 border-blue-500 border-2 ring-blue-500' : ''}`}
            whileHover={{ scale: 1.1, zIndex: 10, backgroundColor: "#9ca3af" }}
            whileTap={{ scale: 0.9, zIndex: 10, transition: { duration: 0.3 } }}
            onClick={() => setSelectedGameMode(prevMode => prevMode === game_mode ? "" : game_mode)}
          >
            <Image
              src={getGameModeIconSrc(game_mode)}
              alt={game_mode}
              width={28}
              height={28}
              unoptimized
              style={{objectFit: "contain"}}
              onError={(e) => {
                e.currentTarget.src = '/brawlstar.png'; 
              }}
            />
          </motion.button>
          ))
        ) : (
          <div className="text-sm text-gray-600">
            {error ? "Backend unavailable" : "Loading maps..."}
          </div>
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
  const { firstPick, setFirstPick, resetEverything, error, rosterMismatch, availableMaps } = useBrawler();

  return (
    <div className='w-full flex flex-col items-center lg:items-stretch lg:flex-row gap-x-12 py-3 gap-y-3 justify-center mb-16 lg:mb-0'>
      <div className='w-full lg:hidden'>
        {error && (
          <div className="mb-3 rounded-xl border border-red-300 bg-red-100 px-4 py-3 text-sm text-red-900">
            {error}
          </div>
        )}
        {!error && availableMaps.length === 0 && (
          <div className="mb-3 rounded-xl border border-yellow-300 bg-yellow-100 px-4 py-3 text-sm text-yellow-900">
            Maps are still loading from the backend.
          </div>
        )}
        {rosterMismatch && (
          <div className="mb-3 rounded-xl border border-yellow-300 bg-yellow-100 px-4 py-3 text-sm text-yellow-900">
            The backend roster and icon mapping do not match yet. Refresh the deployment artifacts before relying on suggestions.
          </div>
        )}
      </div>
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
