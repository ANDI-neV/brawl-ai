"use client";
import { useBrawler } from './brawler-context';

const Selection = () => {

    const {firstPick, setFirstPick} = useBrawler();
    return (
        <button className='flex w-[150px] rounded-xl p-2 justify-center font-bold text-xl'
        onClick={() => setFirstPick(!firstPick)}
        style={{backgroundColor: firstPick ? '#3b82f6' : '#f43f5e'}}
      >
        {firstPick ? 'First Pick' : 'Second Pick'}
      </button>
    )
};

export default Selection;