"use client";
import React, { useEffect } from 'react';
import Image from "next/image";
import { getMapping, Mapping } from './api-handler';
import { useState } from 'react';
import { useBrawler } from './brawler-context';

interface BrawlerIconProps {
  brawler: string;
  isAboveFold?: boolean;
}

export default function BrawlerIcon({ brawler, isAboveFold = false }: BrawlerIconProps) {
  const { loadingMapping, error, brawlerMapping } = useBrawler();

  if (loadingMapping) return <div>Loading...</div>;

  const brawlerId = Object.entries(brawlerMapping).find(([name, id]) => name === brawler)?.[1];
  if (!brawlerId) return <div>Brawler not found</div>;

  return (
    <div className="md:h-[75px] md:w-[75px] h-[60px] w-[60px] border-[4px] md:border-[5px] border-black bg-white bg-opacity-10 p-2 relative rounded-lg overflow-hidden">
      <Image 
        className="rounded-md object-cover object-left"
        src={`https://cdn.brawlify.com/brawlers/borderless/${brawlerId}.png`}
        alt={brawler} 
        fill
        sizes="(max-width: 768px) 75px, 100px"
        priority={isAboveFold}
        style={{
          objectFit: 'cover',
          objectPosition: 'left center'
        }}
      />
      <div className='absolute bottom-0 left-0 right-0 bg-slate-600 bg-opacity-60 text-white text-xs md:text-s rounded-b-xl font-bold text-center p-0.5'> 
        {brawler}
      </div>
    </div>
  );
}