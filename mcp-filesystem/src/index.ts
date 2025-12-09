#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';
import { promises as fs } from 'fs';
import path from 'path';

// 获取允许的目录
function getAllowedDirectories(): string[] {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    return [process.cwd()];
  }
  return args;
}

const allowedDirectories = getAllowedDirectories();

class FilesystemServer {
  private server: Server;

  constructor() {
    this.server = new Server(
      {
        name: 'filesystem-server',
        version: '0.1.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupToolHandlers();
    
    // Error handling
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  private setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'list_allowed_directories',
          description: 'List all directories the server is allowed to access',
          inputSchema: {
            type: 'object',
            properties: {},
            additionalProperties: false,
          },
        },
        {
          name: 'read_text_file',
          description: 'Read complete contents of a file as text',
          inputSchema: {
            type: 'object',
            properties: {
              path: {
                type: 'string',
                description: 'File path to read',
              },
              head: {
                type: 'number',
                description: 'First N lines',
              },
              tail: {
                type: 'number',
                description: 'Last N lines',
              },
            },
            required: ['path'],
            additionalProperties: false,
          },
        },
        {
          name: 'list_directory',
          description: 'List directory contents with [FILE] or [DIR] prefixes',
          inputSchema: {
            type: 'object',
            properties: {
              path: {
                type: 'string',
                description: 'Directory path to list',
              },
            },
            required: ['path'],
            additionalProperties: false,
          },
        },
        {
          name: 'search_files',
          description: 'Recursively search for files/directories that match patterns',
          inputSchema: {
            type: 'object',
            properties: {
              path: {
                type: 'string',
                description: 'Starting directory',
              },
              pattern: {
                type: 'string',
                description: 'Search pattern',
              },
              excludePatterns: {
                type: 'array',
                items: { type: 'string' },
                description: 'Exclude any patterns',
              },
            },
            required: ['path', 'pattern'],
            additionalProperties: false,
          },
        },
        {
          name: 'create_directory',
          description: 'Create new directory or ensure it exists',
          inputSchema: {
            type: 'object',
            properties: {
              path: {
                type: 'string',
                description: 'Directory path to create',
              },
            },
            required: ['path'],
            additionalProperties: false,
          },
        },
        {
          name: 'write_file',
          description: 'Create new file or overwrite existing',
          inputSchema: {
            type: 'object',
            properties: {
              path: {
                type: 'string',
                description: 'File location',
              },
              content: {
                type: 'string',
                description: 'File content',
              },
            },
            required: ['path', 'content'],
            additionalProperties: false,
          },
        },
        {
          name: 'get_file_info',
          description: 'Get detailed file/directory metadata',
          inputSchema: {
            type: 'object',
            properties: {
              path: {
                type: 'string',
                description: 'File or directory path',
              },
            },
            required: ['path'],
            additionalProperties: false,
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      switch (request.params.name) {
        case 'list_allowed_directories':
          return this.handleListAllowedDirectories();
        case 'read_text_file':
          return this.handleReadTextFile(request.params.arguments);
        case 'list_directory':
          return this.handleListDirectory(request.params.arguments);
        case 'search_files':
          return this.handleSearchFiles(request.params.arguments);
        case 'create_directory':
          return this.handleCreateDirectory(request.params.arguments);
        case 'write_file':
          return this.handleWriteFile(request.params.arguments);
        case 'get_file_info':
          return this.handleGetFileInfo(request.params.arguments);
        default:
          throw new McpError(
            ErrorCode.MethodNotFound,
            `Unknown tool: ${request.params.name}`
          );
      }
    });
  }

  private async handleListAllowedDirectories() {
    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(allowedDirectories, null, 2),
        },
      ],
    };
  }

  private async handleReadTextFile(args: any) {
    const { path: filePath, head, tail } = args;
    const resolvedPath = this.resolvePath(filePath);
    
    try {
      let content = await fs.readFile(resolvedPath, 'utf-8');
      
      if (head) {
        const lines = content.split('\n');
        content = lines.slice(0, head).join('\n');
      } else if (tail) {
        const lines = content.split('\n');
        content = lines.slice(-tail).join('\n');
      }

      return {
        content: [
          {
            type: 'text',
            text: content,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error reading file: ${error instanceof Error ? error.message : 'Unknown error'}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async handleListDirectory(args: any) {
    const { path: dirPath } = args;
    const resolvedPath = this.resolvePath(dirPath);
    
    try {
      const entries = await fs.readdir(resolvedPath, { withFileTypes: true });
      const listing = entries.map(entry => 
        `${entry.isDirectory() ? '[DIR]' : '[FILE]'} ${entry.name}`
      ).join('\n');

      return {
        content: [
          {
            type: 'text',
            text: listing,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error listing directory: ${error instanceof Error ? error.message : 'Unknown error'}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async handleSearchFiles(args: any) {
    const { path: searchPath, pattern, excludePatterns = [] } = args;
    const resolvedPath = this.resolvePath(searchPath);
    
    try {
      const matches = await this.searchFilesRecursive(resolvedPath, pattern, excludePatterns);
      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(matches, null, 2),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error searching files: ${error instanceof Error ? error.message : 'Unknown error'}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async handleCreateDirectory(args: any) {
    const { path: dirPath } = args;
    const resolvedPath = this.resolvePath(dirPath);
    
    try {
      await fs.mkdir(resolvedPath, { recursive: true });
      return {
        content: [
          {
            type: 'text',
            text: `Directory created successfully: ${resolvedPath}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error creating directory: ${error instanceof Error ? error.message : 'Unknown error'}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async handleWriteFile(args: any) {
    const { path: filePath, content } = args;
    const resolvedPath = this.resolvePath(filePath);
    
    try {
      await fs.writeFile(resolvedPath, content, 'utf-8');
      return {
        content: [
          {
            type: 'text',
            text: `File written successfully: ${resolvedPath}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error writing file: ${error instanceof Error ? error.message : 'Unknown error'}`,
          },
        ],
        isError: true,
      };
    }
  }

  private async handleGetFileInfo(args: any) {
    const { path: filePath } = args;
    const resolvedPath = this.resolvePath(filePath);
    
    try {
      const stats = await fs.stat(resolvedPath);
      const info = {
        path: resolvedPath,
        size: stats.size,
        isDirectory: stats.isDirectory(),
        isFile: stats.isFile(),
        created: stats.birthtime,
        modified: stats.mtime,
        accessed: stats.atime,
      };

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify(info, null, 2),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: 'text',
            text: `Error getting file info: ${error instanceof Error ? error.message : 'Unknown error'}`,
          },
        ],
        isError: true,
      };
    }
  }

  private resolvePath(requestedPath: string): string {
    const resolved = path.resolve(requestedPath);
    
    // 检查路径是否在允许的目录内
    const isAllowed = allowedDirectories.some(allowedDir => {
      return resolved.startsWith(path.resolve(allowedDir));
    });
    
    if (!isAllowed) {
      throw new McpError(
        ErrorCode.InvalidRequest,
        `Path ${resolved} is not within allowed directories`
      );
    }
    
    return resolved;
  }

  private async searchFilesRecursive(
    dir: string, 
    pattern: string, 
    excludePatterns: string[] = []
  ): Promise<string[]> {
    const matches: string[] = [];
    
    try {
      const entries = await fs.readdir(dir, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        
        // 检查是否匹配排除模式
        const shouldExclude = excludePatterns.some(excludePattern => 
          fullPath.includes(excludePattern) || entry.name.includes(excludePattern)
        );
        
        if (shouldExclude) continue;
        
        if (entry.name.includes(pattern)) {
          matches.push(fullPath);
        }
        
        if (entry.isDirectory()) {
          const subMatches = await this.searchFilesRecursive(fullPath, pattern, excludePatterns);
          matches.push(...subMatches);
        }
      }
    } catch (error) {
      console.error(`Error searching in ${dir}:`, error);
    }
    
    return matches;
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Filesystem MCP server running on stdio');
  }
}

const server = new FilesystemServer();
server.run().catch(console.error);
