"use client";
import { useBrawler } from './brawler-context';
import React from "react";
import {Dropdown, DropdownTrigger, DropdownMenu, DropdownItem, Button} from "@nextui-org/react";
import type { Selection } from "@nextui-org/react";
import { ChevronDown } from "lucide-react";
import { motion } from "framer-motion";

function Menu() {
  const { selectedMap, availableMaps, mapSelectionSetup } = useBrawler();
  const [selectedKeys, setSelectedKeys] = React.useState<Selection>(new Set(selectedMap ? [selectedMap] : []));

  const handleSelectionChange = (keys: Selection) => {
    setSelectedKeys(keys);
    const selected = Array.from(keys)[0] as string;
    mapSelectionSetup(selected);
  };

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
          {availableMaps.map((map) => (
            <DropdownItem key={map} className="text-white hover:bg-gray-700 px-2 py-2 rounded-xl gap-x-2">
              {map}
            </DropdownItem>
          ))}
        </DropdownMenu>
      </Dropdown>
    </div>
  );
}

const Selection = () => {
  const { firstPick, setFirstPick, resetEverything } = useBrawler();

  return (
    <div className='w-full flex flex-col md:flex-row gap-x-12 py-3 items-center gap-y-3 justify-center mb-16 md:mb-0'>
      <Menu />
      <motion.button
        className='flex w-[150px] h-[50px] rounded-xl justify-center items-center p-2 font-bold text-xl'
        onClick={() => setFirstPick(!firstPick)}
        style={{backgroundColor: firstPick ? '#76a8f9' : '#f7798e'}}
        whileHover={{ scale: 1.1, zIndex: 10 }}
        whileTap={{ scale: 0.9, zIndex: 10, transition: { duration: 0.3 } }}
      >
        {firstPick ? 'First Pick' : 'Second Pick'}
      </motion.button>
      <motion.button
        className='flex w-[150px] h-[50px] rounded-xl justify-center items-center p-2 font-bold text-xl'
        onClick={() => resetEverything()}
        style={{backgroundColor: '#9e9191'}}
        whileHover={{ scale: 1.1, zIndex: 10 }}
        whileTap={{ scale: 0.9, zIndex: 10, transition: { duration: 0.3 } }}
      >
        Reset
      </motion.button>
    </div>
  );
};

export default Selection;