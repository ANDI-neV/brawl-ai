"use client"
import { useBrawler } from './brawler-context';
import React, { useCallback } from 'react';
import Image from "next/image";
import { motion } from 'framer-motion';


const BrawlerBans = () => {
    const {brawlerBans, brawlerMapping, removeBrawlerBan} = useBrawler();
    const getBrawlerImageUrl = useCallback((brawler: string) => {
        const brawlerId = Object.entries(brawlerMapping).find(([name, id]) => name === brawler)?.[1];
        return brawlerId 
        ? `https://cdn.brawlify.com/brawlers/borderless/${brawlerId}.png`
        : '/brawlstar.png';
        },
        [brawlerMapping]
    );
        
    

    return (
        <div className="relative items-center justify-center bg-gray-300 rounded-2xl p-2 pt-3">
            <div className="absolute -top-3 left-3 z-20">
                <span className="px-2 py-0.5 bg-yellow-300 text-gray-900 text-sm rounded-xl">Bans</span>
            </div>
            {brawlerBans.length == 0 ?
           <div className='w-20 items-center text-center font-bold text-sm'>
                Right click to ban a brawler
           </div>:
           brawlerBans.map(brawler => (
                <motion.button className='flex flex-row items-center h-20 w-20 mx-auto p-1'
                    onClick={() => removeBrawlerBan(brawler)}
                    whileHover={{ scale: 1.1, zIndex: 10 }}
                    whileTap={{ scale: 0.9, zIndex: 10, transition: { duration: 0.3 } }}
                >
                    <div className="relative w-full h-full">
                    <Image 
                        className="rounded-xl object-cover object-left p-1 bg-gray-800"
                        src={getBrawlerImageUrl(brawler.name)} 
                        alt={brawler.name} 
                        fill
                        sizes="125px"
                        
                        style={{
                            objectFit: 'cover',
                            objectPosition: 'left center',
                            filter: 'grayscale(100%)'
                        }}
                    />
                     <motion.div
                        className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 text-white font-bold text-sm rounded-xl"
                        initial={{ opacity: 0 }}
                        whileHover={{ opacity: 1 }}
                    >
                        Unban
                    </motion.div>
                    </div>
                   
                </motion.button>
           ))
           }
        </div>
    );
};

export default BrawlerBans;