import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createRoot } from 'react-dom/client';
import Webcam from 'react-webcam';
import { motion, AnimatePresence } from 'framer-motion';
import { GoogleGenAI, LiveServerMessage, Modality, Type, Chat } from '@google/genai';
import { 
  Mic, Camera, Power, Activity, Terminal, 
  Wifi, Volume2, Video, Zap, X, Send, 
  RefreshCw, Smartphone, Wrench, Search, VolumeX,
  ChevronRight, ChevronLeft, MessageSquare, Paperclip, FileText, AlertCircle, RotateCcw, AudioWaveform,
  Image as ImageIcon, Sparkles, Play, MapPin, CheckCircle, Scan, MousePointerClick,
  Maximize2, Minimize2, Signal, Cpu, Info, Star, Navigation
} from 'lucide-react';

// --- Configuration ---
const API_KEY = process.env.API_KEY;
const CHAT_MODEL = 'gemini-2.5-flash';
const LIVE_MODEL = 'gemini-2.5-flash-native-audio-preview-09-2025';

const SYSTEM_INSTRUCTION = `You are FixIt Pro, the ultimate intelligent repair assistant.
YOUR IDENTITY:
- You are a professional, knowledgeable expert in mechanics, electronics, home improvement, and DIY repairs.
- You are strictly safety-conscious. ALWAYS warn users about electricity, gas, or pressure hazards before suggesting actions.

YOUR APP CAPABILITIES (Self-Awareness):
1. CHAT MODE: You answer questions, analyze uploaded photos/videos of broken items, and provide step-by-step guides.
2. AR MODE: You use the camera to "see" what the user sees. Users can TAP the screen to highlight specific parts for you to identify.
3. VOICE MODE: You act as a hands-free assistant for users holding tools.
4. EXPERT FINDER: You can search for local professionals (mechanics, plumbers) using Google Maps integration.

BEHAVIOR:
- In CHAT: Be detailed, use bullet points, and provide comprehensive guides.
- In LIVE (AR/Voice): Be concise, conversational, and direct. 
- CRITICAL VISUAL TRIGGER: The user can TAP the screen to highlight an object. When you see a CYAN CIRCLE appear on the video feed, that is a direct COMMAND to identify the object inside it. You MUST SPEAK IMMEDIATELY when the circle appears to describe the object and suggest potential fixes. Do not wait for user audio.
- If you don't know the answer, suggest finding a local expert.

EXPERT FINDER INSTRUCTIONS:
- When asked to find experts, mechanics, or professionals, use the 'googleMaps' tool.
- If you find results, you MUST format them as a JSON array of objects within your text response so the UI can render cards. 
- Format: [{"name": "Name", "rating": 4.5, "specialty": "Mechanic", "address": "123 St", "distance": "0.5 mi"}]
- Do not wrap the JSON in markdown code blocks.`;

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

const ExpertCard: React.FC<{ data: any }> = ({ data }) => (
  <div className="bg-white/5 border border-white/10 p-3 rounded-xl mb-2 flex flex-col gap-1 hover:bg-white/10 transition-colors">
    <div className="flex justify-between items-start">
        <h3 className="font-bold text-yellow-400 text-sm truncate pr-2">{data.name}</h3>
        <div className="flex items-center bg-yellow-500/20 px-1.5 py-0.5 rounded text-[10px] text-yellow-300 gap-1 shrink-0">
            <Star className="w-3 h-3 fill-current" /> {data.rating}
        </div>
    </div>
    <p className="text-xs text-gray-300 font-medium">{data.specialty}</p>
    <div className="flex items-center gap-1.5 text-[11px] text-gray-400 mt-1">
        <MapPin className="w-3 h-3 text-red-400" /> 
        <span className="truncate">{data.address}</span>
    </div>
    {data.distance && (
        <div className="flex items-center gap-1.5 text-[10px] text-blue-300 mt-0.5">
            <Navigation className="w-3 h-3" /> {data.distance} away
        </div>
    )}
  </div>
);

const MessageBubble: React.FC<{ message: Message; onRetry?: (text: string) => void }> = ({ message, onRetry }) => {
  const isUser = message.role === 'user';
  
  // Try to parse structured content (JSON)
  let structuredContent = null;
  if (!isUser && (message.text.trim().startsWith('[') || message.text.trim().startsWith('{') || message.text.includes('['))) {
    try {
        // Attempt to find JSON array in text if mixed
        const jsonMatch = message.text.match(/\[.*\]/s);
        const jsonStr = jsonMatch ? jsonMatch[0] : message.text;
        const parsed = JSON.parse(jsonStr);
        
        if (Array.isArray(parsed)) {
            structuredContent = parsed;
        } else if (parsed.experts && Array.isArray(parsed.experts)) {
            structuredContent = parsed.experts;
        }
    } catch (e) {
        // Failed to parse, treat as normal text
    }
  }

  return (
    <div className={`flex w-full mb-6 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[85%] sm:max-w-[75%] flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
        
        {/* Model Header */}
        {!isUser && (
          <div className="mb-1.5 ml-1 flex items-center gap-1.5">
             <div className="w-5 h-5 rounded-full bg-yellow-500/10 flex items-center justify-center border border-yellow-500/30">
                <Sparkles className="w-3 h-3 text-yellow-500" />
             </div>
             <span className="text-[11px] font-bold tracking-wider text-yellow-500/80">FIXIT PRO</span>
          </div>
        )}

        {/* Attachments */}
        {message.image && (
          <div className={`mb-1 rounded-xl overflow-hidden border border-gray-700 w-64 shadow-lg ${isUser ? 'rounded-br-none' : 'rounded-bl-none'}`}>
             <img src={`data:image/jpeg;base64,${message.image}`} className="w-full" alt="attachment" />
          </div>
        )}
        
        {message.file && !message.image && (
          <div className="mb-2 p-3 bg-[#1E1F20] rounded-xl border border-gray-700 flex items-center gap-3 max-w-full shadow-lg">
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

        <div className={`relative text-[15px] leading-relaxed shadow-xl px-5 py-3.5 ${
          isUser 
            ? 'bg-gradient-to-br from-indigo-600 to-blue-600 text-white rounded-2xl rounded-tr-sm border border-blue-400/20' 
            : 'bg-[#1a1d21] text-gray-100 rounded-2xl rounded-tl-sm border-l-4 border-l-yellow-500 border-y border-r border-white/5'
        }`}>
          {structuredContent ? (
            <div className="flex flex-col">
                <div className="text-sm text-gray-300 mb-3 font-medium">Here are some top-rated experts I found near you:</div>
                {structuredContent.map((expert: any, idx: number) => (
                    <ExpertCard key={idx} data={expert} />
                ))}
            </div>
          ) : (
            <div className="whitespace-pre-wrap">{message.text.replace(/\[.*\]/s, '')}</div>
          )}
          
          {/* Grounding Sources */}
          {message.groundingChunks && message.groundingChunks.length > 0 && (
            <div className="mt-4 flex flex-wrap gap-2 pt-3 border-t border-white/10">
              {message.groundingChunks.map((chunk, idx) => {
                  if (chunk.web) {
                    return (
                      <a 
                        key={idx} 
                        href={chunk.web.uri} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="flex items-center gap-1.5 bg-black/30 hover:bg-black/50 text-[10px] text-blue-300 px-3 py-1.5 rounded-full transition-colors border border-blue-500/20"
                      >
                        <Search className="w-3 h-3" />
                        <span className="truncate max-w-[120px]">{chunk.web.title}</span>
                      </a>
                    );
                  }
                  if (chunk.maps) {
                    return (
                      <a 
                        key={idx} 
                        href={chunk.maps.desktopUri || chunk.maps.uri} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="flex items-center gap-1.5 bg-black/30 hover:bg-black/50 text-[10px] text-yellow-300 px-3 py-1.5 rounded-full transition-colors border border-yellow-500/20"
                      >
                        <MapPin className="w-3 h-3 text-red-400" />
                        <span className="truncate max-w-[120px]">{chunk.maps.title}</span>
                      </a>
                    );
                  }
                  return null;
              })}
            </div>
          )}
          
          <div className={`text-[10px] text-right mt-2 opacity-60 ${isUser ? 'text-blue-100' : 'text-gray-500'}`}>
            {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
        </div>
        
        {message.error && onRetry && (
           <button 
             onClick={() => onRetry(message.text)}
             className="mt-2 text-xs text-red-400 flex items-center gap-1 hover:text-red-300 bg-red-500/10 px-2 py-1 rounded"
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
  const [connectionState, setConnectionState] = useState<'CONNECTING' | 'CONNECTED' | 'DISCONNECTED' | 'ERROR'>('CONNECTING');
  
  // AR Specific State
  const [showGuide, setShowGuide] = useState(mode === 'AR');
  const [arOverlay, setArOverlay] = useState<{
    x: number;
    y: number;
    state: 'SCANNING' | 'ANALYZING' | 'IDENTIFIED';
    text: string;
  } | null>(null);

  // HUD Data
  const [aiConfidence, setAiConfidence] = useState(0);

  const webcamRef = useRef<Webcam>(null);
  const videoIntervalRef = useRef<number | null>(null);
  const silenceIntervalRef = useRef<number | null>(null);
  const aiRef = useRef<GoogleGenAI | null>(null);
  const sessionRef = useRef<any>(null);
  const activeSessionPromiseRef = useRef<Promise<any> | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const wakeLockRef = useRef<any>(null);
  const nextStartTimeRef = useRef<number>(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const logsEndRef = useRef<HTMLDivElement>(null);
  const isCleanedUpRef = useRef(false);
  const awaitingIdentificationRef = useRef(false);
  const lastTapRef = useRef<{x: number, y: number} | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 3;
  
  // Ref to access current overlay inside intervals/callbacks
  const arOverlayRef = useRef<{x: number, y: number, state: string} | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
     arOverlayRef.current = arOverlay;
  }, [arOverlay]);

  useEffect(() => {
    isCleanedUpRef.current = false;
    if (API_KEY) aiRef.current = new GoogleGenAI({ apiKey: API_KEY });
    reconnectAttemptsRef.current = 0;
    startSession();
    requestWakeLock();
    
    // Auto-hide guide after 4 seconds
    if (mode === 'AR') {
      const timer = setTimeout(() => setShowGuide(false), 4000);
      return () => clearTimeout(timer);
    }
    
    return () => {
      cleanup();
      releaseWakeLock();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Wake Lock to prevent mobile devices from sleeping/disconnecting
  const requestWakeLock = async () => {
    try {
      if ('wakeLock' in navigator) {
        wakeLockRef.current = await (navigator as any).wakeLock.request('screen');
      }
    } catch (err) {
      console.log('Wake Lock not supported or denied', err);
    }
  };

  const releaseWakeLock = async () => {
    if (wakeLockRef.current) {
      try {
        await wakeLockRef.current.release();
        wakeLockRef.current = null;
      } catch (err) {
         console.log('Wake Lock release error', err);
      }
    }
  };

  // Updated to aggregate messages if role matches the last one
  const updateLog = (role: 'user' | 'ai', text: string) => {
    setLogs(prev => {
      const lastMsg = prev[prev.length - 1];
      let fullText = text;
      
      let newLogs = [...prev];
      if (lastMsg && lastMsg.role === role) {
        fullText = lastMsg.text + text;
        newLogs[newLogs.length - 1] = { ...lastMsg, text: fullText };
      } else {
        newLogs.push({ id: Math.random().toString(36), role, text });
      }

      // AR Update Logic
      if (role === 'ai' && awaitingIdentificationRef.current) {
          setArOverlay(currentOverlay => {
              if (!currentOverlay) return null;
              // Clean up text for the card
              return { ...currentOverlay, state: 'IDENTIFIED', text: fullText };
          });
      }

      return newLogs;
    });
  };

  const scheduleReconnect = () => {
    if (isCleanedUpRef.current) return;
    
    if (reconnectAttemptsRef.current < maxReconnectAttempts) {
        reconnectAttemptsRef.current += 1;
        setStatus(`Reconnecting (${reconnectAttemptsRef.current}/${maxReconnectAttempts})...`);
        setConnectionState('CONNECTING');
        
        // Exponential backoff: 1s, 2s, 4s...
        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current - 1), 10000);
        
        setTimeout(() => {
            if (!isCleanedUpRef.current) {
                startSession();
            }
        }, delay);
    } else {
        setStatus("Connection Failed");
        setConnectionState('ERROR');
        setAiConfidence(0);
    }
  };

  const startSession = async () => {
    if (!aiRef.current) return;
    cleanup(); 
    isCleanedUpRef.current = false;
    
    if (!window.isSecureContext && window.location.hostname !== 'localhost') {
       setStatus("HTTPS Required");
       setConnectionState('ERROR');
       return;
    }

    if (reconnectAttemptsRef.current === 0) {
        setStatus("Connecting...");
    }
    setConnectionState('CONNECTING');

    try {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      const outputCtx = new AudioContextClass({ sampleRate: 24000 });
      if (outputCtx.state === 'suspended') await outputCtx.resume();
      audioContextRef.current = outputCtx;

      // Attempt to get Mic Stream
      let inputCtx: AudioContext | null = null;
      let stream: MediaStream | null = null;
      
      try {
          inputCtx = new AudioContextClass({ sampleRate: 16000 });
          if (inputCtx.state === 'suspended') await inputCtx.resume();
          inputAudioContextRef.current = inputCtx;

          stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
              channelCount: 1
            } 
          });
      } catch (micErr) {
          console.warn("Mic access denied, proceeding in Vision-Only mode", micErr);
          setStatus("Mic Permission Denied");
      }

      const sessionPromise = aiRef.current.live.connect({
        model: LIVE_MODEL,
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: SYSTEM_INSTRUCTION,
          inputAudioTranscription: {}, 
          outputAudioTranscription: {}, 
        },
        callbacks: {
          onopen: () => {
            setStatus(inputAudioContextRef.current ? "Connected" : "Vision Only Mode");
            setConnectionState('CONNECTED');
            setAiConfidence(95);
            reconnectAttemptsRef.current = 0; // Reset attempts on success
          },
          onmessage: async (msg: LiveServerMessage) => {
            if (msg.serverContent?.outputTranscription?.text) {
               updateLog('ai', msg.serverContent.outputTranscription.text);
            }
            if (msg.serverContent?.inputTranscription?.text) {
               updateLog('user', msg.serverContent.inputTranscription.text);
            }

            if (msg.serverContent?.modelTurn && mode === 'AR') {
                setAiConfidence(100);
                setTimeout(() => { awaitingIdentificationRef.current = false; }, 4000);
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
          onclose: (e) => {
             console.log("Session closed", e);
             if (!isCleanedUpRef.current) {
                scheduleReconnect();
             }
          },
          onerror: (e) => {
            console.error("Session error", e);
            if (!isCleanedUpRef.current) {
                scheduleReconnect();
            }
          }
        }
      });
      
      activeSessionPromiseRef.current = sessionPromise;

      // Only setup audio input processor if stream exists
      if (stream && inputCtx) {
          const source = inputCtx.createMediaStreamSource(stream);
          const processor = inputCtx.createScriptProcessor(2048, 1, 1);

          processor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            
            // Simple VAD / Volume Meter
            let sum = 0;
            for(let i = 0; i < inputData.length; i++) sum += inputData[i] * inputData[i];
            setVolume(Math.sqrt(sum / inputData.length));

            const pcm16 = new Int16Array(inputData.length);
            for (let i = 0; i < inputData.length; i++) pcm16[i] = inputData[i] * 32768;
            
            sessionPromise.then(session => {
               if (activeSessionPromiseRef.current === sessionPromise) {
                 try {
                  session.sendRealtimeInput({
                     media: { mimeType: 'audio/pcm;rate=16000', data: encode(new Uint8Array(pcm16.buffer)) }
                  });
                 } catch (err) {
                   // Ignore send errors if session is closing
                 }
               }
            });
          };

          source.connect(processor);
          processor.connect(inputCtx.destination);
      } else {
          // VISION ONLY: Send Silence to keep session alive and VAD active
          setVolume(0);
          
          const intervalMs = 128; // ~2048 samples at 16000Hz
          const silentBuffer = new Int16Array(2048).fill(0);
          const encodedSilence = encode(new Uint8Array(silentBuffer.buffer));
          
          if (silenceIntervalRef.current) clearInterval(silenceIntervalRef.current);
          
          silenceIntervalRef.current = window.setInterval(() => {
              sessionPromise.then(session => {
                   if (activeSessionPromiseRef.current === sessionPromise) {
                       try {
                           session.sendRealtimeInput({
                               media: { mimeType: 'audio/pcm;rate=16000', data: encodedSilence }
                           });
                       } catch(e) {}
                   }
              });
          }, intervalMs);
      }
      
      if (mode === 'AR') {
         startVideoLoop(sessionPromise);
      }
      
      sessionRef.current = await sessionPromise;
    } catch (e: any) {
      console.error("Connection Setup Error:", e);
      // Ensure we retry even if the initial setup throws (e.g. network down)
      if (!isCleanedUpRef.current && status !== "Mic Permission Denied") {
          scheduleReconnect();
      } else {
          setConnectionState('ERROR');
          setStatus("Connection Failed");
      }
    }
  };

  const startVideoLoop = (sessionPromise: Promise<any>) => {
    if (videoIntervalRef.current) clearInterval(videoIntervalRef.current);
    
    // Increased to ~1.6 FPS (600ms) for better responsiveness
    videoIntervalRef.current = window.setInterval(() => {
      if (activeSessionPromiseRef.current !== sessionPromise) return;
      if (!webcamRef.current) return;
      
      sessionPromise.then(session => {
        if (activeSessionPromiseRef.current !== sessionPromise) return;
        
        try {
            const video = webcamRef.current?.video;
            if (!video || video.videoWidth === 0) return;

            // Use a canvas to draw visual prompts (the "Cyan Circle") onto the feed
            // This ensures the model SEES what the user tapped, without needing text prompts.
            if (!canvasRef.current) {
                canvasRef.current = document.createElement('canvas');
            }
            const canvas = canvasRef.current;
            // Reduced resolution to 384px to save bandwidth while maintaining sufficient detail
            const targetWidth = 384; 
            const scale = targetWidth / video.videoWidth;
            const targetHeight = video.videoHeight * scale;

            if (canvas.width !== targetWidth) {
                canvas.width = targetWidth;
                canvas.height = targetHeight;
            }

            const ctx = canvas.getContext('2d');
            if (ctx) {
                // 1. Draw the raw video frame
                ctx.drawImage(video, 0, 0, targetWidth, targetHeight);

                // 2. Draw Visual Prompt if AR Overlay is active (Scanning)
                if (arOverlayRef.current && arOverlayRef.current.state === 'SCANNING') {
                    // Approximate the tap location on the video frame
                    const x = (arOverlayRef.current.x / window.innerWidth) * targetWidth;
                    const y = (arOverlayRef.current.y / window.innerHeight) * targetHeight;

                    // Draw a bright CYAN target that the model is instructed to look for
                    ctx.strokeStyle = '#00FFFF';
                    ctx.lineWidth = 4;
                    ctx.beginPath();
                    ctx.arc(x, y, 40 * scale, 0, 2 * Math.PI);
                    ctx.stroke();
                    
                    // Draw crosshair center
                    ctx.fillStyle = '#00FFFF';
                    ctx.beginPath();
                    ctx.arc(x, y, 5 * scale, 0, 2 * Math.PI);
                    ctx.fill();
                }

                // 3. Send the composite image with slightly lower quality (0.5) to ensure faster upload
                const base64Data = canvas.toDataURL('image/jpeg', 0.5).split(',')[1];
                session.sendRealtimeInput({
                    media: { mimeType: 'image/jpeg', data: base64Data }
                });
            }

        } catch (err) {
          console.error("Video send error", err);
        }
      });
    }, 600); 
  };

  const handleArTap = (e: React.MouseEvent<HTMLDivElement>) => {
    if (mode !== 'AR') return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Initialize AR Overlay state to 'SCANNING'
    // This triggers the Visual Prompt in startVideoLoop
    setArOverlay({ x, y, state: 'SCANNING', text: '' });
    awaitingIdentificationRef.current = true;
    lastTapRef.current = { x, y };

    // Transition UI to Analyzing state after a short delay
    setTimeout(() => {
        setArOverlay(prev => prev ? { ...prev, state: 'ANALYZING' } : null);
    }, 1500);
    
    // We removed session.send() here because it was causing crashes.
    // Instead, we rely on the strong System Instruction + Visual Prompt (Cyan Circle)
    // to trigger the model's response.
  };

  const cleanup = () => {
    isCleanedUpRef.current = true;
    if (videoIntervalRef.current) clearInterval(videoIntervalRef.current);
    if (silenceIntervalRef.current) clearInterval(silenceIntervalRef.current);
    
    if (inputAudioContextRef.current) {
      inputAudioContextRef.current.close().catch(e => console.error(e));
      inputAudioContextRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(e => console.error(e));
      audioContextRef.current = null;
    }

    sourcesRef.current.forEach(s => {
      try { s.stop(); } catch (e) {}
    });
    sourcesRef.current.clear();
    
    activeSessionPromiseRef.current = null;
    
    if (sessionRef.current) {
        try {
           // @ts-ignore
           if (typeof sessionRef.current.close === 'function') sessionRef.current.close();
        } catch(e) {}
        sessionRef.current = null;
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 bg-[#131314] flex flex-col"
    >
      <div className="absolute inset-0 z-0" onClick={handleArTap}>
        {mode === 'AR' ? (
          <>
             <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                screenshotQuality={0.2}
                videoConstraints={{ facingMode: "environment" }}
                onUserMediaError={(e) => {
                  console.error("Camera Error:", e);
                  setStatus(typeof e === 'string' ? e : "Camera Access Denied");
                }}
                className="w-full h-full object-cover" 
             />
             
             {/* HUD: Connection & Confidence */}
             <div className="absolute top-4 right-4 z-40 flex flex-col items-end gap-2">
                 <div className="flex items-center gap-1.5 bg-black/40 backdrop-blur px-2 py-1 rounded-full border border-white/10">
                     <Signal className={`w-3 h-3 ${connectionState === 'CONNECTED' ? 'text-green-400' : 'text-red-400'}`} />
                     <span className="text-[10px] text-gray-300 font-mono">5G</span>
                 </div>
                 <div className="flex items-center gap-1.5 bg-black/40 backdrop-blur px-2 py-1 rounded-full border border-white/10">
                     <Cpu className="w-3 h-3 text-yellow-400" />
                     <div className="w-12 h-1 bg-gray-700 rounded-full overflow-hidden">
                         <motion.div 
                            className="h-full bg-yellow-400" 
                            animate={{ width: `${aiConfidence}%` }}
                         />
                     </div>
                     <span className="text-[10px] text-gray-300 font-mono">{aiConfidence}%</span>
                 </div>
             </div>

             {/* AR Overlays */}
             <AnimatePresence>
                {/* 1. On-screen Instructions */}
                {showGuide && (
                    <motion.div 
                        initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                        className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 pointer-events-none z-30 flex flex-col items-center text-center p-6 bg-black/60 backdrop-blur-md rounded-2xl border border-yellow-500/30 shadow-2xl max-w-[80%]"
                    >
                        <Scan className="w-12 h-12 text-yellow-500 mb-3 animate-pulse" />
                        <h3 className="text-xl font-bold text-white mb-1">Point & Tap</h3>
                        <p className="text-sm text-gray-200">Align device with the problem area. Tap specific parts to analyze.</p>
                    </motion.div>
                )}

                {/* Unified AR Overlay */}
                {arOverlay && (
                    <div className="absolute inset-0 pointer-events-none z-30">
                        {/* 1. Anchor Point / Reticle */}
                        <motion.div
                            initial={{ scale: 0, opacity: 0 }}
                            animate={{ 
                                scale: 1, 
                                opacity: 1,
                                rotate: arOverlay.state === 'SCANNING' ? 360 : 0
                            }}
                            transition={{ 
                                rotate: { duration: 2, repeat: Infinity, ease: "linear" },
                                scale: { duration: 0.3 }
                            }}
                            style={{ left: arOverlay.x, top: arOverlay.y }}
                            className="absolute -translate-x-1/2 -translate-y-1/2 w-24 h-24 flex items-center justify-center"
                        >
                            {/* Outer Ring */}
                            <div className={`absolute inset-0 border-2 rounded-full border-dashed transition-colors duration-500 ${
                                arOverlay.state === 'IDENTIFIED' ? 'border-green-500/80' : 'border-cyan-400/80'
                            }`} />
                            
                            {/* Inner Brackets */}
                            <div className={`absolute w-16 h-16 transition-all duration-500 ${
                                arOverlay.state === 'IDENTIFIED' ? 'border-green-400' : 'border-cyan-400'
                            }`}>
                                <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-current" />
                                <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-current" />
                                <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-current" />
                                <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-current" />
                            </div>

                            {/* Center Dot (Scanning Phase) */}
                            {arOverlay.state === 'SCANNING' && (
                                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-ping" />
                            )}
                        </motion.div>

                        {/* 2. Info Card (Only when analyzing/identified) */}
                        <AnimatePresence>
                            {arOverlay.state !== 'SCANNING' && (
                                <motion.div
                                    initial={{ opacity: 0, x: 20, y: 20 }}
                                    animate={{ opacity: 1, x: 50, y: -50 }}
                                    exit={{ opacity: 0 }}
                                    style={{ left: arOverlay.x, top: arOverlay.y }}
                                    className="absolute pointer-events-auto"
                                >
                                    {/* Connecting Line */}
                                    <svg className="absolute -left-[50px] top-[50px] w-[50px] h-[50px] pointer-events-none overflow-visible">
                                        <path 
                                            d="M 0,50 L 50,0" 
                                            stroke={arOverlay.state === 'IDENTIFIED' ? '#4ade80' : '#22d3ee'} 
                                            strokeWidth="2" 
                                            fill="none" 
                                        />
                                        <circle cx="0" cy="50" r="3" fill={arOverlay.state === 'IDENTIFIED' ? '#4ade80' : '#22d3ee'} />
                                    </svg>

                                    {/* Card Content */}
                                    <div className={`
                                        w-64 backdrop-blur-md border rounded-xl overflow-hidden shadow-2xl
                                        ${arOverlay.state === 'IDENTIFIED' 
                                            ? 'bg-black/80 border-green-500/30 shadow-green-900/20' 
                                            : 'bg-[#131314]/90 border-cyan-500/30'}
                                    `}>
                                        <div className={`px-4 py-2 text-[10px] font-bold tracking-widest uppercase flex justify-between items-center ${
                                            arOverlay.state === 'IDENTIFIED' ? 'bg-green-500/20 text-green-400' : 'bg-cyan-500/10 text-cyan-400'
                                        }`}>
                                            <span>{arOverlay.state === 'IDENTIFIED' ? 'IDENTIFIED OBJECT' : 'ANALYZING...'}</span>
                                            {arOverlay.state === 'IDENTIFIED' && <CheckCircle className="w-3 h-3" />}
                                        </div>
                                        
                                        <div className="p-4">
                                            {arOverlay.state === 'IDENTIFIED' ? (
                                                <>
                                                    <h3 className="text-lg font-bold text-white mb-1 leading-tight">
                                                        {arOverlay.text.split(/[:\n]/)[0].substring(0, 30)}
                                                    </h3>
                                                    <p className="text-xs text-gray-300 leading-relaxed">
                                                        {arOverlay.text.split(/[:\n](.+)/)[1] || "Retrieving details..."}
                                                    </p>
                                                    <button className="mt-3 w-full py-2 bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg text-[10px] text-gray-300 font-medium transition-colors flex items-center justify-center gap-2">
                                                        <Search className="w-3 h-3" /> FIND REPLACEMENT PARTS
                                                    </button>
                                                </>
                                            ) : (
                                                <div className="flex flex-col gap-2">
                                                    <div className="h-4 bg-white/10 rounded w-3/4 animate-pulse" />
                                                    <div className="h-3 bg-white/5 rounded w-full animate-pulse delay-75" />
                                                    <div className="h-3 bg-white/5 rounded w-5/6 animate-pulse delay-150" />
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>
                    </div>
                )}
             </AnimatePresence>
          </>
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
      
      <div className="relative z-10 p-4 flex items-start gap-3">
         <button 
           onClick={() => {
             onExit(logs);
           }}
           className="bg-[#303134] hover:bg-[#444746] text-white p-2.5 rounded-full backdrop-blur-md transition-all shadow-lg border border-white/10"
         >
           <ChevronLeft className="w-6 h-6" />
         </button>

         <div className={`backdrop-blur border-l-4 p-3 shadow-lg rounded-r-lg flex-1 ${mode === 'AR' ? 'bg-black/60 border-yellow-500' : 'bg-[#1E1F20]/80 border-blue-500'}`}>
            <div className={`flex items-center gap-2 font-bold tracking-widest animate-pulse ${mode === 'AR' ? 'text-yellow-500' : 'text-blue-400'}`}>
               {mode === 'AR' ? <Zap className="w-5 h-5" /> : <AudioWaveform className="w-5 h-5" />} 
               {mode === 'AR' ? 'LIVE VISION' : 'VOICE MODE'}
            </div>
            <div className="flex flex-col">
                <div className="text-[10px] text-gray-400 mt-1 uppercase truncate max-w-[200px]">{status}</div>
                {(connectionState === 'DISCONNECTED' || connectionState === 'ERROR') && (
                   <button 
                      onClick={() => { reconnectAttemptsRef.current = 0; startSession(); }} 
                      className="text-[10px] bg-red-500 hover:bg-red-600 text-white px-3 py-1.5 rounded mt-2 font-bold tracking-wide shadow-lg flex items-center gap-1.5 transition-all w-fit"
                   >
                      <RotateCcw className="w-3 h-3" /> RECONNECT
                   </button>
                )}
            </div>
         </div>
      </div>

      {/* Live Chat Logs - Optimized Styling */}
      <div className="relative z-10 flex-1 flex flex-col justify-end p-4 pb-8 pointer-events-none">
         <div className="pointer-events-auto space-y-4">
             {logs.slice(-2).map((msg) => (
               <motion.div 
                 initial={{ opacity: 0, y: 20 }}
                 animate={{ opacity: 1, y: 0 }}
                 key={msg.id} 
                 className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
               >
                  <div className={`
                    flex flex-col max-w-[85%] px-5 py-3 rounded-2xl backdrop-blur-md shadow-lg border
                    ${msg.role === 'user' 
                      ? 'bg-blue-600/30 border-blue-400/30 text-white rounded-br-none' 
                      : 'bg-[#131314]/60 border-yellow-500/30 text-gray-100 rounded-bl-none'}
                  `}>
                     <span className={`text-[10px] font-bold uppercase mb-1 tracking-widest ${
                        msg.role === 'user' ? 'text-blue-300' : 'text-yellow-500'
                     }`}>
                        {msg.role === 'user' ? 'You' : 'FixIt Pro'}
                     </span>
                     <span className="text-sm leading-relaxed font-medium shadow-black drop-shadow-md">
                        {msg.text}
                     </span>
                  </div>
               </motion.div>
             ))}
             <div ref={logsEndRef} />
         </div>
      </div>

      <div className="relative z-10 p-4 bg-gradient-to-t from-[#131314] to-transparent pointer-events-none">
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
    { id: '0', role: 'model', text: "Hi! How are you? What can I FIXIT for you today? 🛠️\nI'm ready to help you diagnose repairs, analyze videos, or find local experts.", timestamp: Date.now() }
  ]);
  const [input, setInput] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [attachment, setAttachment] = useState<{name: string, type: string, data: string} | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [location, setLocation] = useState<{lat: number, lng: number} | null>(null);
  
  const aiRef = useRef<GoogleGenAI | null>(null);
  const chatSessionRef = useRef<Chat | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Get location for Maps
    navigator.geolocation.getCurrentPosition(
        (pos) => setLocation({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
        (err) => console.log("Location access denied or error:", err)
    );
  }, []);

  // Initialize AI and Chat Session
  useEffect(() => {
    if (API_KEY) {
        aiRef.current = new GoogleGenAI({ apiKey: API_KEY });
        
        // Configure tools - Enable Maps only if location is available
        const tools: any[] = [{ googleSearch: {} }];
        let toolConfig = undefined;

        if (location) {
            tools.push({ googleMaps: {} });
            toolConfig = {
                retrievalConfig: {
                    latLng: {
                        latitude: location.lat,
                        longitude: location.lng
                    }
                }
            };
        }

        chatSessionRef.current = aiRef.current.chats.create({
            model: CHAT_MODEL,
            config: {
                systemInstruction: SYSTEM_INSTRUCTION,
                tools,
                toolConfig
            }
        });
    }
  }, [location]);

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
  
  const handleClear = () => {
      setMessages([]);
      if (aiRef.current) {
          // Re-init session to clear context
          const tools: any[] = [{ googleSearch: {} }];
          let toolConfig = undefined;
          if (location) {
            tools.push({ googleMaps: {} });
            toolConfig = {
                retrievalConfig: {
                    latLng: { latitude: location.lat, longitude: location.lng }
                }
            };
          }
          chatSessionRef.current = aiRef.current.chats.create({
            model: CHAT_MODEL,
            config: {
                systemInstruction: SYSTEM_INSTRUCTION,
                tools,
                toolConfig
            }
        });
      }
      addMessage('model', "Chat cleared! How can I help you now?");
  };

  const handleSend = async (retryText?: string) => {
    const textToSend = retryText || input;
    if ((!textToSend.trim() && !attachment) || !aiRef.current || !chatSessionRef.current) return;
    
    // 1. Optimistic Update (User Message)
    const msgId = Math.random().toString(36).substring(7);
    const newMessage: Message = {
      id: msgId,
      role: 'user',
      text: textToSend,
      timestamp: Date.now(),
      image: attachment?.type.startsWith('image') ? attachment.data : undefined,
      file: attachment ? { name: attachment.name, type: attachment.type, data: attachment.data } : undefined
    };

    setMessages(prev => [...prev, newMessage]);
    setInput('');
    setAttachment(null);
    setIsLoading(true);

    try {
        let responseText = '';
        let groundingChunks = undefined;

        // Construct Message Content
        let messageContent: any = textToSend;
        
        // Handle Multimodal Input
        if (newMessage.image || newMessage.file) {
             const mimeType = newMessage.image ? 'image/jpeg' : newMessage.file?.type || 'application/octet-stream';
             const data = newMessage.image || newMessage.file?.data || '';
             
             messageContent = {
                 parts: [
                     { text: textToSend },
                     { inlineData: { mimeType, data } }
                 ]
             };
        }

        const result = await chatSessionRef.current.sendMessage({ message: messageContent });
        responseText = result.text;
        groundingChunks = result.candidates?.[0]?.groundingMetadata?.groundingChunks;

        setMessages(prev => [...prev, {
            id: Math.random().toString(36).substring(7),
            role: 'model',
            text: responseText,
            timestamp: Date.now(),
            groundingChunks
        }]);

    } catch (error) {
        console.error("Gemini API Error:", error);
        setMessages(prev => [...prev, {
            id: Math.random().toString(36).substring(7),
            role: 'model',
            text: "I'm having trouble connecting to the network or finding that info. Please try again.",
            timestamp: Date.now(),
            error: true
        }]);
    } finally {
        setIsLoading(false);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.onload = (ev) => {
        const base64 = (ev.target?.result as string).split(',')[1];
        setAttachment({
            name: file.name,
            type: file.type,
            data: base64
        });
      };
      reader.readAsDataURL(file);
    }
  };

  const handleLiveExit = (liveLogs: LiveMessage[]) => {
      setMode('CHAT');
      if (liveLogs.length > 0) {
          addMessage('model', "I'm back from Live Mode. Here is a summary of our session.");
          // Optionally append live logs to chat
      }
  };

  const handleFindExpert = () => {
    handleSend("Find the best local repair experts near me");
  };

  return (
    <div className="flex flex-col h-screen bg-[#131314] text-white font-sans overflow-hidden">
      {/* Background Scanline Effect */}
      <div className="fixed inset-0 pointer-events-none opacity-5 z-0">
          <div className="scan-line" />
      </div>

      {mode !== 'CHAT' && (
          <LiveSession mode={mode} onExit={handleLiveExit} />
      )}

      {/* Header */}
      <header className="flex-none p-4 border-b border-white/10 bg-[#1e1f20]/80 backdrop-blur-md z-10 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-yellow-500 to-amber-600 flex items-center justify-center shadow-lg shadow-yellow-900/20">
            <Wrench className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-lg tracking-tight leading-none">FixIt Pro</h1>
            <div className="flex items-center gap-1.5 mt-0.5">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                </span>
                <span className="text-[10px] text-gray-400 uppercase tracking-widest font-medium">AI Agent Active</span>
            </div>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
           <button 
             onClick={handleClear} 
             className="p-2 hover:bg-white/5 rounded-full text-gray-400 hover:text-white transition-colors"
             title="Clear Chat"
           >
              <RefreshCw className="w-5 h-5" />
           </button>
        </div>
      </header>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-4 scrollbar-hide">
         <div className="max-w-3xl mx-auto flex flex-col pt-2">
            {messages.map((msg) => (
               <MessageBubble key={msg.id} message={msg} onRetry={handleSend} />
            ))}
            {isLoading && (
               <div className="flex justify-start mb-6">
                  <div className="flex items-center gap-2 px-4 py-3 bg-[#1a1d21] rounded-2xl rounded-tl-sm border border-white/5">
                     <div className="flex gap-1">
                        <div className="w-2 h-2 bg-yellow-500 rounded-full animate-bounce" style={{ animationDelay: '0s' }} />
                        <div className="w-2 h-2 bg-yellow-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                        <div className="w-2 h-2 bg-yellow-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
                     </div>
                  </div>
               </div>
            )}
            <div ref={messagesEndRef} />
         </div>
      </div>

      {/* Input Area */}
      <div className="flex-none p-4 bg-[#1e1f20] border-t border-white/10 z-20">
         <div className="max-w-3xl mx-auto flex flex-col gap-3">
            
            {/* Attachment Preview */}
            {attachment && (
                <div className="flex items-center gap-3 bg-[#2a2b2d] p-2 rounded-lg border border-white/10 w-fit pr-4 animate-in fade-in slide-in-from-bottom-2">
                    <div className="w-10 h-10 bg-black/30 rounded flex items-center justify-center">
                        {attachment.type.startsWith('image') ? (
                            <img src={`data:${attachment.type};base64,${attachment.data}`} className="w-full h-full object-cover rounded" />
                        ) : (
                            <FileText className="w-5 h-5 text-gray-400" />
                        )}
                    </div>
                    <div className="flex flex-col">
                        <span className="text-xs text-gray-200 truncate max-w-[150px] font-medium">{attachment.name}</span>
                        <span className="text-[10px] text-gray-500 uppercase">{attachment.type.split('/')[1]}</span>
                    </div>
                    <button 
                        onClick={() => setAttachment(null)}
                        className="ml-2 p-1 hover:bg-white/10 rounded-full text-gray-400 hover:text-red-400"
                    >
                        <X className="w-4 h-4" />
                    </button>
                </div>
            )}

            <div className="flex items-end gap-2">
                {/* Unified Mode Toggles (Visible on all screens now) */}
                <div className="flex items-center gap-1 bg-[#131314] p-1 rounded-xl border border-white/5 mr-2">
                   <button 
                      onClick={() => setMode('CHAT')}
                      className={`p-2 rounded-lg transition-all ${mode === 'CHAT' ? 'bg-white/10 text-white' : 'text-gray-500 hover:text-gray-300'}`}
                      title="Chat Mode"
                   >
                      <MessageSquare className="w-5 h-5" />
                   </button>
                   <button 
                      onClick={() => setMode('VOICE')}
                      className={`p-2 rounded-lg transition-all ${mode === 'VOICE' ? 'bg-blue-500/20 text-blue-400' : 'text-gray-500 hover:text-blue-400'}`}
                      title="Voice Mode"
                   >
                      <Mic className="w-5 h-5" />
                   </button>
                   <button 
                      onClick={() => setMode('AR')}
                      className={`p-2 rounded-lg transition-all ${mode === 'AR' ? 'bg-yellow-500/20 text-yellow-400' : 'text-gray-500 hover:text-yellow-400'}`}
                      title="AR Vision Mode"
                   >
                      <Camera className="w-5 h-5" />
                   </button>
                </div>

                {/* Main Input */}
                <div className="flex-1 bg-[#131314] rounded-2xl border border-white/10 focus-within:border-yellow-500/50 focus-within:ring-1 focus-within:ring-yellow-500/20 transition-all flex items-center p-1.5 shadow-inner">
                    <button 
                        onClick={() => fileInputRef.current?.click()}
                        className="p-2 text-gray-400 hover:text-white hover:bg-white/5 rounded-xl transition-colors"
                        title="Attach File"
                    >
                        <Paperclip className="w-5 h-5" />
                    </button>
                    <input 
                        type="file" 
                        ref={fileInputRef} 
                        className="hidden" 
                        accept="image/*,video/*"
                        onChange={handleFileSelect}
                    />

                    {/* Find Expert Button */}
                    <button 
                        onClick={handleFindExpert}
                        className="p-2 text-gray-400 hover:text-yellow-400 hover:bg-white/5 rounded-xl transition-colors"
                        title="Find Experts Nearby"
                    >
                        <MapPin className="w-5 h-5" />
                    </button>
                    
                    <textarea 
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handleSend();
                            }
                        }}
                        placeholder="Describe the issue..."
                        className="flex-1 bg-transparent text-sm text-white placeholder-gray-500 px-2 py-2.5 outline-none resize-none max-h-32 scrollbar-hide"
                        rows={1}
                    />
                    
                    <button 
                        onClick={() => handleSend()}
                        disabled={!input.trim() && !attachment}
                        className={`p-2 rounded-xl transition-all duration-200 ${
                            (input.trim() || attachment) 
                                ? 'bg-yellow-500 text-black hover:bg-yellow-400 shadow-[0_0_15px_rgba(234,179,8,0.3)]' 
                                : 'bg-white/5 text-gray-600'
                        }`}
                    >
                        <Send className="w-5 h-5" />
                    </button>
                </div>
            </div>
         </div>
      </div>
    </div>
  );
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
