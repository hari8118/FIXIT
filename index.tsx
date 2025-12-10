import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import Webcam from 'react-webcam';
import { motion, AnimatePresence } from 'framer-motion';
import { GoogleGenAI, LiveServerMessage, Modality } from '@google/genai';
import { 
  Mic, Camera, Power, Activity, Terminal, 
  Wifi, Volume2, Video, Zap, X, Send, 
  RefreshCw, Smartphone, Wrench, Search, VolumeX,
  ChevronRight, MessageSquare, Paperclip, FileText, AlertCircle, RotateCcw, AudioWaveform,
  Image as ImageIcon, Sparkles, Play
} from 'lucide-react';

// --- Configuration ---
const API_KEY = process.env.API_KEY;
const CHAT_MODEL = 'gemini-2.5-flash';
const LIVE_MODEL = 'gemini-2.5-flash-native-audio-preview-09-2025';

// --- Types ---
type AppMode = 'CHAT' | 'AR' | 'VOICE';

interface Message {
  id: string;
  role: 'user' | 'model';
  text: string;
  image?: string;
  file?: { name: string; type: string; data: string }; // data is base64
  timestamp: number;
  groundingChunks?: any[];
  error?: boolean;
}

interface LiveMessage {
  id: string;
  role: 'user' | 'ai';
  text: string;
}

// --- Audio Helper Functions (Live API) ---
function encode(bytes: Uint8Array) {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

// --- Components ---

const MessageBubble: React.FC<{ message: Message; onRetry?: (text: string) => void }> = ({ message, onRetry }) => {
  const isUser = message.role === 'user';
  return (
    <div className={`flex w-full mb-6 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[90%] sm:max-w-[80%] flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
        
        {/* Model Icon */}
        {!isUser && (
          <div className="mb-2 flex items-center gap-2 text-yellow-500">
             <Sparkles className="w-5 h-5" />
             <span className="text-sm font-semibold text-gray-300">FixIt Pro</span>
          </div>
        )}

        {/* Image Attachment */}
        {message.image && (
          <div className="mb-2 rounded-xl overflow-hidden border border-gray-700 w-64">
             <img src={`data:image/jpeg;base64,${message.image}`} className="w-full" alt="attachment" />
          </div>
        )}
        
        {/* File/Video Attachment */}
        {message.file && !message.image && (
          <div className="mb-2 p-3 bg-[#1E1F20] rounded-xl border border-gray-700 flex items-center gap-3 max-w-full">
             {message.file.type.startsWith('video') ? (
                <div className="flex items-center gap-3">
                   <div className="w-10 h-10 rounded-full bg-red-900/50 flex items-center justify-center border border-red-500/30">
                      <Play className="w-5 h-5 text-red-400 fill-current" />
                   </div>
                   <div className="text-sm">
                      <div className="font-bold text-gray-200 truncate max-w-[150px]">Video Analysis</div>
                      <div className="text-xs text-gray-500 uppercase">{message.file.type.split('/')[1]} • VIDEO</div>
                   </div>
                </div>
             ) : (
                <>
                   <FileText className="w-8 h-8 text-yellow-500" />
                   <div className="text-sm">
                      <div className="font-bold text-gray-200 truncate max-w-[150px]">{message.file.name}</div>
                      <div className="text-xs text-gray-500 uppercase">{message.file.type.split('/')[1]}</div>
                   </div>
                </>
             )}
          </div>
        )}

        <div className={`relative text-[15px] leading-7 ${
          isUser 
            ? 'bg-[#282A2C] text-gray-100 rounded-[20px] px-5 py-3' 
            : 'text-gray-100 pl-0'
        }`}>
          <div className="whitespace-pre-wrap">{message.text}</div>
          
          {/* Grounding Sources */}
          {message.groundingChunks && message.groundingChunks.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {message.groundingChunks.map((chunk, idx) => {
                  if (chunk.web) {
                    return (
                      <a 
                        key={idx} 
                        href={chunk.web.uri} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="flex items-center gap-1.5 bg-[#1E1F20] hover:bg-[#303134] text-xs text-gray-300 px-3 py-1.5 rounded-full border border-gray-700 transition-colors"
                      >
                        <Search className="w-3 h-3" />
                        <span className="truncate max-w-[150px]">{chunk.web.title}</span>
                      </a>
                    );
                  }
                  return null;
              })}
            </div>
          )}
        </div>
        
        {message.error && onRetry && (
           <button 
             onClick={() => onRetry(message.text)}
             className="mt-2 text-xs text-red-400 flex items-center gap-1 hover:text-red-300"
           >
             <RotateCcw className="w-3 h-3" /> Retry Send
           </button>
        )}
      </div>
    </div>
  );
};

// --- Live Session Interface (AR & Voice) ---
const LiveSession = ({ mode, onExit }: { mode: 'AR' | 'VOICE'; onExit: (logs: LiveMessage[]) => void }) => {
  const [logs, setLogs] = useState<LiveMessage[]>([]);
  const [volume, setVolume] = useState(0);
  const [status, setStatus] = useState('Initializing...');
  
  const webcamRef = useRef<Webcam>(null);
  const videoIntervalRef = useRef<number | null>(null);
  const aiRef = useRef<GoogleGenAI | null>(null);
  const sessionRef = useRef<any>(null);
  const activeSessionPromiseRef = useRef<Promise<any> | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (API_KEY) aiRef.current = new GoogleGenAI({ apiKey: API_KEY });
    startSession();
    return () => cleanup();
  }, []);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Updated to aggregate messages if role matches the last one
  const updateLog = (role: 'user' | 'ai', text: string) => {
    setLogs(prev => {
      const lastMsg = prev[prev.length - 1];
      if (lastMsg && lastMsg.role === role) {
        const newLogs = [...prev];
        newLogs[newLogs.length - 1] = { ...lastMsg, text: lastMsg.text + text };
        return newLogs;
      }
      // Allow shorter start messages but filter absolute empty/whitespace
      if (!text.trim()) return prev;
      return [...prev, { id: Math.random().toString(36), role, text }];
    });
  };

  const startSession = async () => {
    if (!aiRef.current) return;
    try {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      const inputCtx = new AudioContextClass({ sampleRate: 16000 });
      const outputCtx = new AudioContextClass({ sampleRate: 24000 });
      audioContextRef.current = outputCtx;

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const source = inputCtx.createMediaStreamSource(stream);
      // Optimized: Reduced buffer size to 2048 (approx 128ms latency) for faster response
      const processor = inputCtx.createScriptProcessor(2048, 1, 1);
      
      const sessionPromise = aiRef.current.live.connect({
        model: LIVE_MODEL,
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: mode === 'AR' 
             ? "You are FixIt Pro. RAPIDLY identify parts/issues in the video feed. Be extremely concise. Focus on fixing." 
             : "You are FixIt Pro. Listen for mechanical faults. Be concise and direct.",
          inputAudioTranscription: {}, 
          outputAudioTranscription: {}, 
        },
        callbacks: {
          onopen: () => {
            setStatus("Connected");
          },
          onmessage: async (msg: LiveServerMessage) => {
            if (msg.serverContent?.outputTranscription?.text) {
               updateLog('ai', msg.serverContent.outputTranscription.text);
            }
            if (msg.serverContent?.inputTranscription?.text) {
               updateLog('user', msg.serverContent.inputTranscription.text);
            }

            const audioData = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData) {
               const ctx = audioContextRef.current;
               if (!ctx) return;
               nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
               const buffer = await decodeAudioData(decode(audioData), ctx, 24000, 1);
               const sourceNode = ctx.createBufferSource();
               sourceNode.buffer = buffer;
               sourceNode.connect(ctx.destination);
               sourceNode.addEventListener('ended', () => {
                 sourcesRef.current.delete(sourceNode);
               });
               sourceNode.start(nextStartTimeRef.current);
               nextStartTimeRef.current += buffer.duration;
               sourcesRef.current.add(sourceNode);
            }
          },
          onclose: () => setStatus("Disconnected"),
          onerror: (e) => {
            console.error(e);
            setStatus("Error");
          }
        }
      });
      
      activeSessionPromiseRef.current = sessionPromise;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        
        let sum = 0;
        for(let i = 0; i < inputData.length; i++) sum += inputData[i] * inputData[i];
        setVolume(Math.sqrt(sum / inputData.length));

        const pcm16 = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) pcm16[i] = inputData[i] * 32768;
        
        // Critical: Send audio only when session is ready using the promise
        sessionPromise.then(session => {
           if (activeSessionPromiseRef.current === sessionPromise) {
              session.sendRealtimeInput({
                 media: { mimeType: 'audio/pcm;rate=16000', data: encode(new Uint8Array(pcm16.buffer)) }
              });
           }
        });
      };

      source.connect(processor);
      processor.connect(inputCtx.destination);
      
      if (mode === 'AR') {
         startVideoLoop(sessionPromise);
      }
      
      sessionRef.current = await sessionPromise;
    } catch (e) {
      console.error(e);
      setStatus("Failed");
    }
  };

  const startVideoLoop = (sessionPromise: Promise<any>) => {
    if (videoIntervalRef.current) clearInterval(videoIntervalRef.current);
    
    // Optimized: 200ms (5 FPS) for faster AI response, but low res to keep stable
    videoIntervalRef.current = window.setInterval(() => {
      // Check if session is still valid
      if (activeSessionPromiseRef.current !== sessionPromise) return;
      if (!webcamRef.current) return;
      
      sessionPromise.then(session => {
        if (activeSessionPromiseRef.current !== sessionPromise) return;
        
        try {
          // Optimized: 256px width + 0.15 quality = Tiny payload for max speed/stability
          // We must calculate height to satisfy ScreenshotDimensions type { width: number, height: number }
          let screenshotConfig: { width: number; height: number } | undefined;
          const video = webcamRef.current?.video;
          
          if (video && video.videoWidth > 0 && video.videoHeight > 0) {
            const ratio = video.videoHeight / video.videoWidth;
            screenshotConfig = { width: 256, height: Math.round(256 * ratio) };
          }
          
          const imageSrc = webcamRef.current?.getScreenshot(screenshotConfig);

          if (imageSrc) {
            session.sendRealtimeInput({
              media: { mimeType: 'image/jpeg', data: imageSrc.split(',')[1] }
            });
          }
        } catch (err) {
          console.error("Failed to send video frame:", err);
        }
      });
    }, 200); 
  };

  const cleanup = () => {
    if (videoIntervalRef.current) clearInterval(videoIntervalRef.current);
    if (audioContextRef.current) audioContextRef.current.close();
    sourcesRef.current.forEach(s => s.stop());
    // Invalidate active session promise to stop loops
    activeSessionPromiseRef.current = null;
    
    if (sessionRef.current) {
        try {
           // @ts-ignore
           if (typeof sessionRef.current.close === 'function') sessionRef.current.close();
        } catch(e) {}
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 bg-[#131314] flex flex-col"
    >
      <div className="absolute inset-0 z-0">
        {mode === 'AR' ? (
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            screenshotQuality={0.15}
            videoConstraints={{ facingMode: "environment" }}
            className="w-full h-full object-cover" 
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-[#131314] relative overflow-hidden">
             {[...Array(3)].map((_, i) => (
                <motion.div
                  key={i}
                  animate={{ scale: [1, 1.5 + volume * 5, 1], opacity: [0.3, 0, 0.3] }}
                  transition={{ duration: 2, repeat: Infinity, delay: i * 0.4, ease: "easeInOut" }}
                  className="absolute w-64 h-64 rounded-full border border-yellow-500/20 bg-yellow-500/5"
                />
             ))}
             <Volume2 className="w-24 h-24 text-yellow-500 opacity-80 z-10" />
          </div>
        )}
      </div>
      
      <div className="relative z-10 p-4 flex justify-between items-start">
         <div className={`backdrop-blur border-l-4 p-3 shadow-lg rounded-r-lg ${mode === 'AR' ? 'bg-black/60 border-yellow-500' : 'bg-[#1E1F20]/80 border-blue-500'}`}>
            <div className={`flex items-center gap-2 font-bold tracking-widest animate-pulse ${mode === 'AR' ? 'text-yellow-500' : 'text-blue-400'}`}>
               {mode === 'AR' ? <Zap className="w-5 h-5" /> : <AudioWaveform className="w-5 h-5" />} 
               {mode === 'AR' ? 'LIVE VISION' : 'VOICE MODE'}
            </div>
            <div className="text-[10px] text-gray-400 mt-1 uppercase">{status}</div>
         </div>
         <button 
           onClick={() => {
             onExit(logs);
           }}
           className="bg-[#303134] hover:bg-[#444746] text-white p-3 rounded-full backdrop-blur-md transition-all shadow-lg"
         >
           <X className="w-6 h-6" />
         </button>
      </div>

      {/* Live Chat Logs - Optimized Styling */}
      <div className="relative z-10 flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-white/20">
         {logs.map((msg) => (
           <div key={msg.id} className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`
                flex flex-col max-w-[85%] px-4 py-3 rounded-2xl backdrop-blur-md shadow-lg
                ${msg.role === 'user' 
                  ? 'bg-blue-600/60 border border-blue-400/30 text-white rounded-tr-sm' 
                  : 'bg-black/90 border border-yellow-500 text-yellow-400 font-mono rounded-tl-sm shadow-[0_0_15px_rgba(234,179,8,0.3)]'}
              `}>
                 <span className={`text-[10px] font-bold uppercase mb-1 tracking-widest ${
                    msg.role === 'user' ? 'text-blue-200' : 'text-yellow-600'
                 }`}>
                    {msg.role === 'user' ? 'You' : 'FixIt Pro'}
                 </span>
                 <span className="text-sm leading-relaxed font-medium">
                    {msg.text}
                 </span>
              </div>
           </div>
         ))}
         <div ref={logsEndRef} />
      </div>

      <div className="relative z-10 p-4 bg-gradient-to-t from-[#131314] to-transparent">
         <div className="flex items-center justify-center gap-2 text-xs text-gray-500 font-mono">
            <Activity className="w-3 h-3" /> AUDIO LEVEL: {(volume * 100).toFixed(0)}%
         </div>
      </div>
    </motion.div>
  );
};

// --- Main App ---
const App = () => {
  const [mode, setMode] = useState<AppMode>('CHAT');
  const [messages, setMessages] = useState<Message[]>([
    { id: '0', role: 'model', text: "Systems Online. Attach videos, manuals, or use Live modes to diagnose your device.", timestamp: Date.now() }
  ]);
  const [input, setInput] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [attachment, setAttachment] = useState<{name: string, type: string, data: string} | null>(null);
  
  const aiRef = useRef<GoogleGenAI | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (API_KEY) aiRef.current = new GoogleGenAI({ apiKey: API_KEY });
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const addMessage = (role: 'user' | 'model', text: string, extra: Partial<Message> = {}) => {
    setMessages(prev => [...prev, {
      id: Math.random().toString(36).substring(7),
      role,
      text,
      timestamp: Date.now(),
      ...extra
    }]);
  };

  const handleSend = async (retryText?: string) => {
    const textToSend = retryText || input;
    if ((!textToSend.trim() && !attachment) || !aiRef.current) return;
    
    // Optimistic Update
    const msgId = Math.random().toString(36).substring(7);
    const newMessage: Message = {
      id: msgId,
      role: 'user',
      text: textToSend,
      timestamp: Date.now(),
      image: attachment?.type.startsWith('image') ? attachment.data : undefined,
      file: attachment && !attachment.type.startsWith('image') ? attachment : undefined
    };

    setMessages(prev => [...prev, newMessage]);
    if (!retryText) {
      setInput('');
      setAttachment(null);
    }

    try {
      const parts: any[] = [{ text: textToSend }];
      
      if (newMessage.image) {
         parts.push({ inlineData: { mimeType: 'image/jpeg', data: newMessage.image } });
      } else if (newMessage.file) {
         parts.push({ inlineData: { mimeType: newMessage.file.type, data: newMessage.file.data } });
         
         if (newMessage.file.type.startsWith('video')) {
            parts.push({ text: "Analyze this video for mechanical issues, strange noises, or visual defects. Provide a step-by-step diagnosis." });
         } else {
            parts.push({ text: "Use the attached file to answer questions if relevant." });
         }
      }

      const response = await aiRef.current.models.generateContent({
        model: CHAT_MODEL,
        contents: [{ role: 'user', parts: parts }],
        config: { tools: [{ googleSearch: {} }] }
      });
      
      const text = response.text || "No response received.";
      const grounding = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
      addMessage('model', text, { groundingChunks: grounding });
    } catch (e) {
      setMessages(prev => prev.map(m => m.id === msgId ? { ...m, error: true } : m));
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.size > 20 * 1024 * 1024) {
      alert("File too large. Please upload videos under 20MB.");
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
       const base64 = (reader.result as string).split(',')[1];
       setAttachment({
          name: file.name,
          type: file.type,
          data: base64
       });
    };
    reader.readAsDataURL(file);
  };

  const toggleVoice = () => {
    if (isListening) return; 
    const SpeechRecognition = (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) return alert("Speech API not supported");

    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);
    recognition.onresult = (e: any) => setInput(prev => (prev + ' ' + e.results[0][0].transcript).trim());
    recognition.start();
  };

  return (
    <div className="h-screen bg-[#131314] text-[#E3E3E3] font-sans overflow-hidden flex flex-col">
      <AnimatePresence>
        {(mode === 'AR' || mode === 'VOICE') && (
          <LiveSession mode={mode} onExit={(sessionLogs) => {
             setMode('CHAT');
             // Append session logs as individual messages
             sessionLogs.forEach(log => {
                if (log.text.trim()) {
                   addMessage(log.role === 'user' ? 'user' : 'model', log.text);
                }
             });
          }} />
        )}
      </AnimatePresence>

      {/* Header */}
      <header className="bg-[#131314] p-4 flex justify-between items-center z-10 border-b border-white/5">
        <div className="flex items-center gap-2">
           <Wrench className="w-6 h-6 text-yellow-500"/>
           <h1 className="font-semibold text-xl tracking-tight text-white">FixIt Pro</h1>
        </div>
        <button onClick={() => setMessages([])} className="p-2 text-gray-400 hover:text-white hover:bg-[#303134] rounded-full transition-colors">
           <RefreshCw className="w-5 h-5" />
        </button>
      </header>

      {/* Chat Area */}
      <main className="flex-1 overflow-y-auto p-4 scrollbar-thin scrollbar-thumb-gray-800">
         <div className="max-w-3xl mx-auto w-full">
            {messages.map(m => (
              <MessageBubble 
                key={m.id} 
                message={m} 
                onRetry={(text) => handleSend(text)} 
              />
            ))}
            <div ref={messagesEndRef} />
         </div>
      </main>

      {/* Footer / Input Area */}
      <footer className="bg-[#131314] p-4 pb-6 border-t border-white/5">
         <div className="max-w-3xl mx-auto w-full flex flex-col gap-3">
            
            {/* Mode Chips */}
            <div className="flex items-center gap-2 px-1">
               <button 
                 onClick={() => setMode('AR')}
                 className="flex items-center gap-2 px-4 py-2 rounded-full border border-[#444746] hover:bg-[#303134] text-sm font-medium text-gray-200 transition-colors"
               >
                  <Camera className="w-4 h-4 text-yellow-500" />
                  AR Mode
               </button>
               <button 
                 onClick={() => setMode('VOICE')}
                 className="flex items-center gap-2 px-4 py-2 rounded-full border border-[#444746] hover:bg-[#303134] text-sm font-medium text-gray-200 transition-colors"
               >
                  <AudioWaveform className="w-4 h-4 text-blue-400" />
                  Voice
               </button>
            </div>
            
            {/* Attachment Preview */}
            {attachment && (
              <div className="mx-2 flex items-center gap-3 bg-[#1E1F20] p-3 rounded-2xl w-fit border border-gray-700">
                 {attachment.type.startsWith('image') ? (
                    <img src={`data:${attachment.type};base64,${attachment.data}`} className="h-10 w-10 object-cover rounded-lg" alt="preview" />
                 ) : attachment.type.startsWith('video') ? (
                    <div className="h-10 w-10 bg-red-900/40 rounded-lg flex items-center justify-center border border-red-500/20">
                       <Play className="w-5 h-5 text-red-400 fill-current" />
                    </div>
                 ) : (
                    <FileText className="h-10 w-10 text-yellow-500" />
                 )}
                 <div className="flex flex-col">
                    <span className="text-xs truncate max-w-[150px] text-gray-300 font-medium">{attachment.name}</span>
                    <span className="text-[10px] text-gray-500 uppercase">{attachment.type.split('/')[1]}</span>
                 </div>
                 <button onClick={() => setAttachment(null)} className="text-gray-400 hover:text-white"><X className="w-4 h-4"/></button>
              </div>
            )}

            {/* Input Bar */}
            <div className="relative bg-[#1E1F20] rounded-[28px] flex items-center p-2 pl-4 pr-2 transition-colors focus-within:bg-[#303134]">
               
               <input 
                 type="file" 
                 ref={fileInputRef} 
                 style={{ display: 'none' }}
                 className="hidden" 
                 onChange={handleFileSelect} 
                 accept="image/*,application/pdf,video/*"
               />
               
               {/* Left: Attachment */}
               <button 
                 onClick={() => fileInputRef.current?.click()}
                 className="p-2 text-gray-400 hover:text-white bg-[#303134] hover:bg-[#444746] rounded-full transition-colors mr-2"
                 title="Attach File/Video"
               >
                 <Paperclip className="w-5 h-5" />
               </button>

               {/* Center: Input */}
               <textarea 
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSend()}
                  placeholder="Ask a question..."
                  className="flex-1 bg-transparent border-none focus:ring-0 resize-none h-12 py-3 text-base text-white placeholder-gray-500"
               />

               <div className="flex items-center gap-2">
                 {/* Right: Mic */}
                 <button 
                   onClick={toggleVoice} 
                   className={`p-3 rounded-full transition-colors ${isListening ? 'bg-red-500/20 text-red-500 animate-pulse' : 'text-gray-400 hover:text-white hover:bg-[#444746]'}`}
                 >
                    <Mic className="w-5 h-5" />
                 </button>

                 {/* Far Right: Send */}
                 {(input.trim() || attachment) && (
                   <button 
                      onClick={() => handleSend()}
                      className="p-3 bg-yellow-500 hover:bg-yellow-400 text-black rounded-full transition-transform active:scale-95"
                   >
                      <Send className="w-5 h-5" />
                   </button>
                 )}
               </div>
            </div>
         </div>
      </footer>
    </div>
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);