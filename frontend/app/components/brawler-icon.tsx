"use client";
import React from 'react';
import Image from 'next/image';

interface BrawlerIconProps {
  brawler: string;
}

export default function BrawlerIcon({ brawler }: BrawlerIconProps) {
  const brawler_image_url = `/brawler_images/${brawler}.png`;

  return (
    <div className="md:h-[100px] md:w-[100px] h-[75px] w-[75px] border-2 bg-white bg-opacity-10 p-2 relative rounded-xl overflow-hidden">
      <div className="w-full h-full relative">
        <Image 
          className='rounded-xl object-cover object-left'
          src={brawler_image_url} 
          alt={brawler} 
          layout="fill"
          objectFit="cover"
          objectPosition="left center"
        />
      </div>
      <div className='absolute bottom-0 left-0 right-0 bg-white bg-opacity-20 text-white text-xs rounded-b-xl text-center p-1'> 
        {brawler}
      </div>
    </div>
  );

}