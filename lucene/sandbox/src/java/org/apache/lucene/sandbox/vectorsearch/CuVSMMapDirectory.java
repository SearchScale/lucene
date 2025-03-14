/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.sandbox.vectorsearch;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.apache.lucene.store.MMapDirectory;

/**
 * A MMapDirectory implementation that excludes fsyncing CuVS files (because cuVS has already written them to the right place).
 */
public class CuVSMMapDirectory extends MMapDirectory {

  public CuVSMMapDirectory(Path path) throws IOException {
    super(path);
  }

  @Override
  public void deleteFile(String name) throws IOException {
    System.out.println("Delete called for: " + name);
    super.deleteFile(name);
  }
  @Override
  protected void fsync(String name) throws IOException {
    if (name.endsWith(".vcag")) {
      System.out.println("Ignoring writing " + name);
      String fname = directory.resolve(name).toString();
      
      System.out.println(fname + " Size: " + new File(fname).length());
      System.out.println(fname +".tmpcuvs Size: " + new File(fname + ".copyme").length());
      
      if (new File(fname).exists()) new File(fname).delete();
      Files.move(Paths.get(fname + ".tmpcuvs"), Paths.get(fname));
      return;
    }
    
    super.fsync(name);
  }
}
