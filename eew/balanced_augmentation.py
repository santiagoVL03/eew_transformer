"""
Balanced augmentation for imbalanced seismic datasets.

This module provides augmentation that targets only the minority class (noise)
to balance the dataset and improve model performance.
"""

import numpy as np
from .augmentation import WaveformAugmenter
from .advanced_augmentation import AdvancedWaveformAugmenter


class BalancedAugmenter:
    """
    Augmenter que aplica augmentación SOLO a la clase minoritaria (ruido).
    
    Esto ayuda a balancear datasets desbalanceados donde hay muchos más
    terremotos que señales de ruido.
    
    NOTA: Este augmenter aplica transformaciones pero NO crea copias múltiples.
    Para sobremuestreo real, usar NoiseOversamplingDataset.
    """
    
    # Estadísticas
    _stats = {
        'earthquakes_seen': 0,
        'noise_seen': 0,
        'earthquakes_augmented': 0,
        'noise_augmented': 0,
        'total_noise_augmentations': 0
    }
    
    def __init__(
        self,
        augment_noise_only=True,
        use_advanced=True,
        **augmenter_kwargs
    ):
        """
        Args:
            augment_noise_only: Si True, solo augmenta señales de ruido
            use_advanced: Si True, usa AdvancedWaveformAugmenter, sino WaveformAugmenter
            **augmenter_kwargs: Argumentos para el augmenter base
        """
        self.augment_noise_only = augment_noise_only
        self.use_advanced = use_advanced
        
        # Filtrar kwargs según el tipo de augmenter
        # AdvancedWaveformAugmenter acepta: channel_dropout, channel_drop_prob, baseline_drift
        # WaveformAugmenter solo acepta: add_noise, noise_snr_range, scale_amplitude, time_shift, sampling_rate, p
        
        if use_advanced:
            # AdvancedWaveformAugmenter acepta todos los kwargs
            self.augmenter = AdvancedWaveformAugmenter(
                track_stats=False,  # Nosotros manejamos las stats
                **augmenter_kwargs
            )
        else:
            # Filtrar solo los kwargs que WaveformAugmenter acepta
            basic_kwargs = {}
            allowed_keys = {'add_noise', 'noise_snr_range', 'scale_amplitude', 'time_shift', 'sampling_rate', 'p'}
            for key, value in augmenter_kwargs.items():
                if key in allowed_keys:
                    basic_kwargs[key] = value
            
            self.augmenter = WaveformAugmenter(
                track_stats=False,
                **basic_kwargs
            )
    
    def __call__(self, waveform, label):
        """
        Aplica augmentación basada en la etiqueta.
        
        Args:
            waveform: Forma de onda (channels, samples)
            label: Etiqueta (1 = terremoto, 0 = ruido)
        
        Returns:
            waveform augmentado, label
        """
        # Rastrear estadísticas
        if label == 1:
            BalancedAugmenter._stats['earthquakes_seen'] += 1
        else:
            BalancedAugmenter._stats['noise_seen'] += 1
        
        # Si es terremoto y solo queremos augmentar ruido, retornar sin cambios
        if self.augment_noise_only and label == 1:
            return waveform, label
        
        # Si es ruido, aplicar augmentación
        if label == 0:
            BalancedAugmenter._stats['noise_augmented'] += 1
            BalancedAugmenter._stats['total_noise_augmentations'] += 1
            
            augmented = self.augmenter(waveform)
            return augmented, label
        
        # Terremotos (si augment_noise_only es False)
        BalancedAugmenter._stats['earthquakes_augmented'] += 1
        return self.augmenter(waveform), label
    
    @classmethod
    def print_stats(cls):
        """Imprime estadísticas de augmentación balanceada."""
        stats = cls._stats
        total_seen = stats['earthquakes_seen'] + stats['noise_seen']
        
        if total_seen == 0:
            print("\n⚠️  No se han procesado señales con BalancedAugmenter aún")
            return
        
        print("\n" + "="*60)
        print("ESTADÍSTICAS DE AUGMENTACIÓN BALANCEADA")
        print("="*60)
        print(f"Señales procesadas:")
        print(f"  Terremotos vistos:      {stats['earthquakes_seen']:,} ({stats['earthquakes_seen']/total_seen*100:.2f}%)")
        print(f"  Ruido visto:            {stats['noise_seen']:,} ({stats['noise_seen']/total_seen*100:.2f}%)")
        print(f"\nAugmentaciones aplicadas:")
        print(f"  Terremotos augmentados: {stats['earthquakes_augmented']:,}")
        print(f"  Ruido augmentado:       {stats['noise_augmented']:,}")
        print(f"  Total aug. de ruido:    {stats['total_noise_augmentations']:,}")
        
        if stats['noise_seen'] > 0:
            print(f"\nFactor de augmentación de ruido: {stats['total_noise_augmentations']/stats['noise_seen']:.2f}x")
        
        print("\n⚠️  NOTA: Para sobremuestreo real (crear múltiples copias),")
        print("   usar NoiseOversamplingDataset en lugar de BalancedAugmenter")
        print("="*60 + "\n")
    
    @classmethod
    def reset_stats(cls):
        """Reinicia estadísticas."""
        cls._stats = {
            'earthquakes_seen': 0,
            'noise_seen': 0,
            'earthquakes_augmented': 0,
            'noise_augmented': 0,
            'total_noise_augmentations': 0
        }


class NoiseOversamplingDataset:
    """
    Wrapper para un dataset que sobremuestrea la clase de ruido.
    
    Esto crea copias virtuales de las señales de ruido con augmentación
    para balancear el dataset.
    
    IMPORTANTE: Esta es la clase que debes usar para realmente balancear el dataset.
    """
    
    # Estadísticas
    _stats = {
        'original_earthquakes': 0,
        'original_noise': 0,
        'virtual_earthquakes': 0,
        'virtual_noise': 0,
        'oversampling_ratio': 0.0
    }
    
    def __init__(self, base_dataset, oversampling_ratio=4.0, augmenter=None):
        """
        Args:
            base_dataset: Dataset base (STEADDataset)
            oversampling_ratio: Factor de sobremuestreo para ruido (ej: 4.0 = crear 4 copias de cada ruido)
            augmenter: Augmenter a aplicar (debe aceptar waveform y label)
        """
        self.base_dataset = base_dataset
        self.oversampling_ratio = oversampling_ratio
        self.augmenter = augmenter
        
        # Crear índices virtuales
        self._create_virtual_indices()
    
    def _create_virtual_indices(self):
        """Crea índices virtuales con sobremuestreo de ruido."""
        self.virtual_indices = []
        self.virtual_labels = []
        
        # Contadores
        n_earthquakes = 0
        n_noise = 0
        
        # Obtener todas las muestras del dataset base
        n_samples = len(self.base_dataset)
        
        print(f"\n{'='*60}")
        print("CREANDO DATASET CON SOBREMUESTREO DE RUIDO")
        print(f"{'='*60}")
        print(f"Procesando {n_samples:,} muestras del dataset base...")
        
        # Determinar cómo acceder a las etiquetas
        has_labels = hasattr(self.base_dataset, 'labels') and self.base_dataset.labels is not None
        is_lazy = hasattr(self.base_dataset, 'lazy_load') and self.base_dataset.lazy_load
        
        for idx in range(n_samples):
            label = None
            
            # Intentar obtener etiqueta
            if has_labels:
                label = self.base_dataset.labels[idx]
            elif is_lazy and hasattr(self.base_dataset, 'seisbench_dataset') and hasattr(self.base_dataset, 'indices'):
                # En modo lazy, obtener del metadata
                try:
                    dataset_idx = self.base_dataset.indices[idx]
                    metadata = self.base_dataset.seisbench_dataset.metadata.iloc[dataset_idx]
                    trace_category = metadata.get('trace_category', 'earthquake_local')
                    label = 1 if 'earthquake' in str(trace_category).lower() else 0
                except Exception:
                    # Si falla, cargar el sample para obtener la etiqueta
                    try:
                        _, sample_label = self.base_dataset[idx]
                        if hasattr(sample_label, 'item'):
                            label = int(sample_label.item())
                        else:
                            label = int(sample_label)
                    except Exception:
                        label = None
            
            # Si no pudimos obtener la etiqueta, intentar cargando el sample
            if label is None:
                try:
                    _, sample_label = self.base_dataset[idx]
                    if hasattr(sample_label, 'item'):
                        label = int(sample_label.item())
                    else:
                        label = int(sample_label)
                except Exception:
                    # Fallback: asumir que es terremoto
                    label = 1
            
            # Contar original
            if label == 1:
                n_earthquakes += 1
            else:
                n_noise += 1
            
            # Agregar el índice original
            self.virtual_indices.append(idx)
            self.virtual_labels.append(label)
            
            # Si es ruido (label=0), agregar copias adicionales
            if label == 0:
                n_copies = int(self.oversampling_ratio) - 1  # -1 porque ya agregamos el original
                for _ in range(n_copies):
                    self.virtual_indices.append(idx)
                    self.virtual_labels.append(label)
        
        # Actualizar estadísticas
        NoiseOversamplingDataset._stats['original_earthquakes'] = n_earthquakes
        NoiseOversamplingDataset._stats['original_noise'] = n_noise
        NoiseOversamplingDataset._stats['oversampling_ratio'] = self.oversampling_ratio
        
        # Contar virtuales
        virtual_earthquakes = sum(1 for l in self.virtual_labels if l == 1)
        virtual_noise = sum(1 for l in self.virtual_labels if l == 0)
        
        NoiseOversamplingDataset._stats['virtual_earthquakes'] = virtual_earthquakes
        NoiseOversamplingDataset._stats['virtual_noise'] = virtual_noise
        
        # Imprimir resultados
        total_original = n_earthquakes + n_noise
        total_virtual = len(self.virtual_indices)
        
        print(f"\n📊 DISTRIBUCIÓN ORIGINAL:")
        print(f"  Terremotos: {n_earthquakes:,} ({n_earthquakes/total_original*100:.2f}%)")
        print(f"  Ruido:      {n_noise:,} ({n_noise/total_original*100:.2f}%)")
        print(f"  Total:      {total_original:,}")
        if n_noise > 0:
            print(f"  Ratio:      {n_earthquakes/n_noise:.2f}:1 (terremoto:ruido)")
        else:
            print(f"  Ratio:      N/A (no hay ruido)")
        
        print(f"\n📊 DISTRIBUCIÓN DESPUÉS DEL SOBREMUESTREO ({self.oversampling_ratio}x ruido):")
        print(f"  Terremotos: {virtual_earthquakes:,} ({virtual_earthquakes/total_virtual*100:.2f}%)")
        print(f"  Ruido:      {virtual_noise:,} ({virtual_noise/total_virtual*100:.2f}%)")
        print(f"  Total:      {total_virtual:,}")
        if virtual_noise > 0:
            print(f"  Ratio:      {virtual_earthquakes/virtual_noise:.2f}:1 (terremoto:ruido)")
        else:
            print(f"  Ratio:      N/A (no hay ruido después del sobremuestreo)")
        
        print(f"\n✅ MEJORA:")
        print(f"  Ruido aumentado de {n_noise:,} a {virtual_noise:,} (+{virtual_noise-n_noise:,} muestras)")
        print(f"  Balance mejorado de {n_noise/total_original*100:.2f}% a {virtual_noise/total_virtual*100:.2f}%")
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.virtual_indices)
    
    def __getitem__(self, idx):
        # Obtener índice real
        real_idx = self.virtual_indices[idx]
        
        # Obtener datos del dataset base
        waveform, label = self.base_dataset[real_idx]
        
        # Aplicar augmentación si está disponible
        if self.augmenter is not None:
            # El augmenter debe manejar numpy arrays, convertir de torch si es necesario
            import torch
            if isinstance(waveform, torch.Tensor):
                waveform_np = waveform.numpy()
                waveform_aug, label_val = self.augmenter(waveform_np, label.item() if isinstance(label, torch.Tensor) else label)
                waveform = torch.FloatTensor(waveform_aug)
                label = torch.FloatTensor([label_val])
            else:
                waveform, label = self.augmenter(waveform, label)
        
        return waveform, label
    
    @classmethod
    def print_stats(cls):
        """Imprime estadísticas del sobremuestreo."""
        stats = cls._stats
        
        if stats['original_earthquakes'] == 0 and stats['original_noise'] == 0:
            print("\n⚠️  No se ha creado ningún NoiseOversamplingDataset aún")
            return
        
        total_original = stats['original_earthquakes'] + stats['original_noise']
        total_virtual = stats['virtual_earthquakes'] + stats['virtual_noise']
        
        print("\n" + "="*60)
        print("ESTADÍSTICAS DE SOBREMUESTREO DE RUIDO")
        print("="*60)
        print(f"Dataset Original:")
        print(f"  Terremotos: {stats['original_earthquakes']:,} ({stats['original_earthquakes']/total_original*100:.2f}%)")
        print(f"  Ruido:      {stats['original_noise']:,} ({stats['original_noise']/total_original*100:.2f}%)")
        
        print(f"\nDataset con Sobremuestreo ({stats['oversampling_ratio']}x):")
        print(f"  Terremotos: {stats['virtual_earthquakes']:,} ({stats['virtual_earthquakes']/total_virtual*100:.2f}%)")
        print(f"  Ruido:      {stats['virtual_noise']:,} ({stats['virtual_noise']/total_virtual*100:.2f}%)")
        
        print(f"\nMejora:")
        print(f"  +{stats['virtual_noise']-stats['original_noise']:,} muestras de ruido")
        print(f"  Balance: {stats['original_noise']/total_original*100:.2f}% → {stats['virtual_noise']/total_virtual*100:.2f}%")
        print("="*60 + "\n")
    
    @classmethod
    def reset_stats(cls):
        """Reinicia estadísticas."""
        cls._stats = {
            'original_earthquakes': 0,
            'original_noise': 0,
            'virtual_earthquakes': 0,
            'virtual_noise': 0,
            'oversampling_ratio': 0.0
        }
